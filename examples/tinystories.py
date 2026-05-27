"""
X-MoE Trainer Example

A minimal example showing how to train a Mixture-of-Experts Transformer
using the Trainer API. Just define your dataset, build the model, and train.
"""

from datasets import load_dataset
from x_transformers import TransformerWrapper, Decoder

from x_moe import MoETransformerWrapper, Trainer
from x_moe.data import TextDataset, train_tokenizer


def main():
    print("1. Load data")
    ds = load_dataset("roneneldan/TinyStories")
    train_texts = ds["train"]["text"][:10000]
    val_texts = ds["validation"]["text"][:10000]

    print("2. Train or load a tokenizer")
    tokenizer = train_tokenizer(
        train_texts,
        vocab_size=129280,
        save_path="tokenizer.json",
    )

    print("3. Create datasets (with caching to avoid re-tokenizing each epoch)")
    train_ds = TextDataset(train_texts, tokenizer, max_seq_len=256, cache=True)
    val_ds = TextDataset(val_texts, tokenizer, max_seq_len=256, cache=True)

    print("4. Build the model")
    decoder = Decoder(
        dim=256,
        depth=12,
        heads=8,
        ff_glu=True,
        ff_mult=4,
        ff_dropout=0.1,
        rotary_pos_emb=True,
        ff_no_bias=True,
    )

    transformer = TransformerWrapper(
        num_tokens=tokenizer.get_vocab_size(),
        max_seq_len=256,
        attn_layers=decoder,
        emb_dropout=0.1,
        tie_embedding=True,
        use_abs_pos_emb=False,
    )

    model = MoETransformerWrapper(
        transformer=transformer,
        num_experts=32,
        expert_top_k=2,
        routing_strategy="top_k",
        load_balance_loss_weight=0.01,
        z_loss_weight=1e-4,
        moe_every_n_layers=1,
        glu=True,
        mult=4,
        dropout=0.1,
        no_bias=True,
        zero_init_output=True,
        batched_experts=True,
        max_batch_size=16,
        use_hca=True,
        sigmoid_routing=True
    )

    print("5. Train with the Trainer")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=10,
        batch_size=16,
        gradient_accumulate=4,
        lr=3e-4,
        optimizer="muon",
        aux_loss_every=4,
        pad_to_max=True,
        num_workers=8,
    )
    trainer.train(validation_string="Once upon a time")

    print("6. Save checkpoint")
    trainer.save()

    print("7. Load from checkpoint and chat")
    trainer.release()
    trainer = Trainer.load("checkpoints/best_model.pt", tokenizer=tokenizer)
    print(trainer.chat("Once upon a time"))


if __name__ == "__main__":
    main()
