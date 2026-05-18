def main():
    import torch
    from x_transformers import TransformerWrapper, Decoder
    from easy_moe import (
        MoETransformerWrapper,
        HCA,
        HybridAttentionBlock,
        MuonWithAdamW,
        configure_muon_optimizer,
    )

    # Build a MoE Transformer with optional HCA/CSA compressed attention
    decoder = Decoder(
        dim=256,
        depth=6,
        heads=8,
        ff_glu=True,
        ff_mult=4,
        ff_dropout=0.1,
        attn_dropout=0.1,
        rotary_pos_emb=True,
        ff_no_bias=True,
    )

    transformer = TransformerWrapper(
        num_tokens=256,
        max_seq_len=512,
        attn_layers=decoder,
        emb_dropout=0.1,
        tie_embedding=True,
        use_abs_pos_emb=False,
    )

    # Optional: attach a HybridAttentionBlock (HCA + CSA) as an additional module
    # This can be used for long-context compression alongside the standard attention
    ds4_block = HybridAttentionBlock(
        dim=256,
        hca_config={"kv_dim": 64, "num_query_heads": 4, "compression_rate": 4, "window_size": 16},
    )

    model = MoETransformerWrapper(
        transformer=transformer,
        num_experts=8,
        expert_top_k=2,
        routing_strategy="top_k",
        moe_every_n_layers=2,
        glu=True,
        mult=4,
        dropout=0.1,
        no_bias=True,
        #ds4_attention=ds4_block,
    )

    print(f"MoE Transformer + HCA parameters: {model.num_params:,}")

    x = torch.randint(0, 256, (2, 64))
    loss = model(x)
    print(f"Training loss: {loss.item():.4f}")
    print(f"MoE auxiliary loss: {model.moe_aux_loss.item():.4f}")

    # Test Muon optimizer
    muon_opt, adamw_opt = configure_muon_optimizer(model, lr=1e-3, adamw_lr=3e-4)
    optimizer = MuonWithAdamW(muon_opt, adamw_opt)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print("Muon optimizer step completed")

    # Test generation
    model.eval()
    prompt = torch.randint(0, 256, (1, 10))
    output = model.generate(prompt, seq_len=32, temperature=0.8)
    print(f"Generated shape: {output.shape}")


if __name__ == "__main__":
    main()