import torch
import os
import math
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForCausalLM
from bitsandbytes.optim import Adam8bit
from datasets import load_dataset
from tqdm import tqdm
import datetime
import aiohttp

class Config:
    """Training configuration"""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16
    MD_REVISION = "2024-07-23"
    EPOCHS = 1
    BATCH_SIZE = 1
    GRAD_ACCUM_STEPS = 1
    LR = 1e-5
    USE_WANDB = False
    ANSWER_EOS = "<|endoftext|>"
    IMG_TOKENS = 729
    CACHE_DIR = "data2/"
    MAX_SAMPLES = 100  # Limit to first 100 samples

class CaptchaDataset(Dataset):
    """Dataset loader for Google DOCCI dataset"""
    def __init__(self, split='train', max_samples=None):
        self.data = load_dataset(
            "google/docci",
            trust_remote_code=True,
            cache_dir=Config.CACHE_DIR,
            storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}}
        )[split]
        
        # Limit dataset size if max_samples is specified
        if max_samples is not None:
            self.data = self.data.select(range(min(max_samples, len(self.data))))
            print(f"Dataset limited to {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "image": sample["image"],
            "qa": [{"question": "Describe this image.", "answer": sample["description"]}]
        }

class MoondreamTrainer:
    def __init__(self):
        self._setup_environment()
        self.model, self.tokenizer = self._initialize_model()
        self.datasets = self._load_datasets()
        self.dataloaders = self._create_dataloaders()
        self.optimizer = self._setup_optimizer()
        self.total_steps = Config.EPOCHS * len(self.dataloaders["train"]) // Config.GRAD_ACCUM_STEPS

    def _setup_environment(self):
        """Setup CUDA environment"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    def _initialize_model(self):
        """Initialize model and tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision=Config.MD_REVISION)
        model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision=Config.MD_REVISION,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if Config.DEVICE == "cuda" else None,
            torch_dtype=Config.DTYPE,
            device_map={"": Config.DEVICE}
        )
        
        # Enable training modes and gradient checkpointing
        model.vision_encoder.train()
        model.text_model.train()
        model.text_model.transformer.gradient_checkpointing_enable()
        return model, tokenizer

    def _load_datasets(self):
        """Load datasets"""
        return {"train": CaptchaDataset("train", max_samples=Config.MAX_SAMPLES)}

    def _create_dataloaders(self):
        """Create data loaders"""
        return {
            "train": DataLoader(
                self.datasets["train"],
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                collate_fn=self._collate_fn
            )
        }

    def _setup_optimizer(self):
        """Setup optimizer with different learning rates for vision and text models"""
        return Adam8bit(
            [
                {"params": self.model.text_model.parameters()},
                {"params": self.model.vision_encoder.parameters(), "lr": Config.LR * 0.1}
            ],
            lr=Config.LR * 0.1,
            betas=(0.9, 0.95),
            eps=1e-6
        )

    def _collate_fn(self, batch):
        """Collate batch of samples"""
        images = [sample['image'] for sample in batch]
        images = [self.model.vision_encoder.preprocess(image) for image in images]

        labels_acc = []
        tokens_acc = []

        for sample in batch:
            toks = [self.tokenizer.bos_token_id]
            labs = [-100] * (Config.IMG_TOKENS + 1)

            for qa in sample['qa']:
                q_t = self.tokenizer(
                    f"\n\nQuestion: {qa['question']}\n\nAnswer:",
                    add_special_tokens=False
                ).input_ids
                toks.extend(q_t)
                labs.extend([-100] * len(q_t))

                a_t = self.tokenizer(
                    f" {qa['answer']}{Config.ANSWER_EOS}",
                    add_special_tokens=False
                ).input_ids
                toks.extend(a_t)
                labs.extend(a_t)

            tokens_acc.append(toks)
            labels_acc.append(labs)

        max_len = max(len(labels) for labels in labels_acc)
        attn_mask_acc = []

        for i in range(len(batch)):
            len_i = len(labels_acc[i])
            pad_i = max_len - len_i

            labels_acc[i].extend([-100] * pad_i)
            tokens_acc[i].extend([self.tokenizer.eos_token_id] * pad_i)
            attn_mask_acc.append([1] * len_i + [0] * pad_i)

        return (
            images,
            torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc]),
            torch.stack([torch.tensor(l, dtype=torch.long) for l in labels_acc]),
            torch.stack([torch.tensor(a, dtype=torch.bool) for a in attn_mask_acc]),
        )

    def _compute_loss(self, batch):
        """Compute loss for a batch"""
        images, tokens, labels, attn_mask = batch
        tokens = tokens.to(Config.DEVICE)
        labels = labels.to(Config.DEVICE)
        attn_mask = attn_mask.to(Config.DEVICE)
        
        img_embs = self.model.vision_encoder(images)
        tok_embs = self.model.text_model.get_input_embeddings()(tokens)
        inputs_embeds = torch.cat((tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1)
        
        outputs = self.model.text_model(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attn_mask,
        )
        return outputs.loss

    def _lr_schedule(self, step):
        """Calculate learning rate based on current step"""
        x = step / self.total_steps
        if x < 0.1:
            return 0.1 * Config.LR + 0.9 * Config.LR * x / 0.1
        return 0.1 * Config.LR + 0.9 * Config.LR * (1 + math.cos(math.pi * (x - 0.1))) / 2

    def train(self):
        """Main training loop"""
        step = 0
        for epoch in range(Config.EPOCHS):
            print(f"\nTraining on {len(self.datasets['train'])} samples")
            for batch in tqdm(self.dataloaders["train"], desc=f"Epoch {epoch + 1}/{Config.EPOCHS}"):
                step += 1
                loss = self._compute_loss(batch)
                loss.backward()
                
                if step % Config.GRAD_ACCUM_STEPS == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    lr = self._lr_schedule(step / Config.GRAD_ACCUM_STEPS)
                    self.optimizer.param_groups[0]['lr'] = lr  # Text model
                    self.optimizer.param_groups[1]['lr'] = lr * 0.1  # Vision encoder

    def save_model(self):
        """Save the trained model"""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        save_path = f"checkpoints/moondream-ducci-dataset-with-vision-encoder-ft-{timestamp}"
        self.model.save_pretrained(save_path)
        return save_path

def main():
    trainer = MoondreamTrainer()
    trainer.train()
    save_path = trainer.save_model()
    print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    main()