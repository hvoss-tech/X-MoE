import pytest
import torch
import torch.nn as nn

from x_transformers import TransformerWrapper, Decoder

from x_moe import (
    MoETransformerWrapper,
    MoEFFN,
    HCA,
    CSA,
    DS4AttentionLayer,
    HybridAttentionBlock,
    Muon,
    MuonWithAdamW,
    configure_muon_optimizer,
    replace_ffn_with_moe,
    collect_moe_aux_loss,
    reset_moe_aux_loss,
)


class TestMoEWrapper:
    def test_basic_forward(self):
        decoder = Decoder(dim=64, depth=2, heads=4, ff_glu=True, rotary_pos_emb=True)
        transformer = TransformerWrapper(num_tokens=100, max_seq_len=64, attn_layers=decoder)
        model = MoETransformerWrapper(
            transformer=transformer, num_experts=4, expert_top_k=2,
            glu=True, mult=4, no_bias=True,
        )
        x = torch.randint(0, 100, (2, 32))
        loss = model(x)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_generate(self):
        decoder = Decoder(dim=64, depth=2, heads=4, ff_glu=True, rotary_pos_emb=True)
        transformer = TransformerWrapper(num_tokens=100, max_seq_len=64, attn_layers=decoder)
        model = MoETransformerWrapper(
            transformer=transformer, num_experts=4, expert_top_k=2,
            glu=True, mult=4, no_bias=True,
        )
        model.eval()
        prompt = torch.randint(0, 100, (1, 5))
        out = model.generate(prompt, seq_len=10, temperature=0.8)
        assert out.shape == (1, 10)

    def test_aux_loss(self):
        decoder = Decoder(dim=64, depth=2, heads=4, ff_glu=True, rotary_pos_emb=True)
        transformer = TransformerWrapper(num_tokens=100, max_seq_len=64, attn_layers=decoder)
        model = MoETransformerWrapper(
            transformer=transformer, num_experts=4, expert_top_k=2,
            glu=True, mult=4, no_bias=True,
        )
        x = torch.randint(0, 100, (2, 32))
        model(x)
        aux = model.moe_aux_loss
        assert aux.item() >= 0
        model.reset_moe_aux_loss()
        assert model.moe_aux_loss.item() == 0.0

    def test_expert_choice(self):
        decoder = Decoder(dim=64, depth=2, heads=4, ff_glu=True, rotary_pos_emb=True)
        transformer = TransformerWrapper(num_tokens=100, max_seq_len=64, attn_layers=decoder)
        model = MoETransformerWrapper(
            transformer=transformer, num_experts=4, expert_top_k=2,
            routing_strategy="expert_choice", capacity_factor=1.0,
            glu=True, mult=4, no_bias=True,
        )
        x = torch.randint(0, 100, (2, 32))
        loss = model(x)
        assert not torch.isnan(loss)

    def test_moe_layers_param(self):
        decoder = Decoder(dim=64, depth=4, heads=4, ff_glu=True, rotary_pos_emb=True)
        transformer = TransformerWrapper(num_tokens=100, max_seq_len=64, attn_layers=decoder)
        model = MoETransformerWrapper(
            transformer=transformer, num_experts=4, expert_top_k=2,
            moe_layers=[0, 3], glu=True, mult=4, no_bias=True,
        )
        moe_count = sum(1 for m in model.modules() if isinstance(m, MoEFFN))
        assert moe_count == 2

    def test_backward(self):
        decoder = Decoder(dim=64, depth=2, heads=4, ff_glu=True, rotary_pos_emb=True)
        transformer = TransformerWrapper(num_tokens=100, max_seq_len=64, attn_layers=decoder)
        model = MoETransformerWrapper(
            transformer=transformer, num_experts=4, expert_top_k=2,
            glu=True, mult=4, no_bias=True,
        )
        x = torch.randint(0, 100, (2, 32))
        loss = model(x)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(not torch.isnan(g).any() for g in grads)


class TestDS4Integration:
    def test_wrapper_with_ds4(self):
        from x_moe.attention import HybridAttentionBlock
        decoder = Decoder(dim=64, depth=2, heads=4, ff_glu=True, rotary_pos_emb=True)
        transformer = TransformerWrapper(num_tokens=100, max_seq_len=64, attn_layers=decoder)
        ds4_block = HybridAttentionBlock(
            dim=64,
            hca_config={"kv_dim": 32, "num_query_heads": 4, "compression_rate": 4, "window_size": 0},
        )
        model = MoETransformerWrapper(
            transformer=transformer, num_experts=4, expert_top_k=2,
            glu=True, mult=4, no_bias=True,
            ds4_attention=ds4_block,
        )
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)

    def test_ds4_with_moe_backward(self):
        from x_moe.attention import HybridAttentionBlock
        decoder = Decoder(dim=64, depth=2, heads=4, ff_glu=True, rotary_pos_emb=True)
        transformer = TransformerWrapper(num_tokens=100, max_seq_len=64, attn_layers=decoder)
        ds4_block = HybridAttentionBlock(
            dim=64,
            hca_config={"kv_dim": 32, "num_query_heads": 4, "compression_rate": 4, "window_size": 0},
        )
        model = MoETransformerWrapper(
            transformer=transformer, num_experts=4, expert_top_k=2,
            glu=True, mult=4, no_bias=True,
            ds4_attention=ds4_block,
        )
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        loss.backward()
        has_valid_grads = any(p.grad is not None and not torch.isnan(p.grad).any()
                              for p in model.parameters())
        assert has_valid_grads


class TestOptimizerIntegration:
    def test_muon_training_step(self):
        decoder = Decoder(dim=64, depth=2, heads=4, ff_glu=True, rotary_pos_emb=True)
        transformer = TransformerWrapper(num_tokens=100, max_seq_len=64, attn_layers=decoder)
        model = MoETransformerWrapper(
            transformer=transformer, num_experts=4, expert_top_k=2,
            glu=True, mult=4, no_bias=True,
        )
        muon_opt, adamw_opt = configure_muon_optimizer(model, lr=1e-3)
        combo = MuonWithAdamW(muon_opt, adamw_opt)

        x = torch.randint(0, 100, (2, 16))
        for _ in range(3):
            loss = model(x)
            model.reset_moe_aux_loss()
            loss.backward()
            combo.step()
            combo.zero_grad()

    def test_full_training_loop(self):
        decoder = Decoder(dim=64, depth=2, heads=4, ff_glu=True, rotary_pos_emb=True)
        transformer = TransformerWrapper(num_tokens=100, max_seq_len=64, attn_layers=decoder)
        model = MoETransformerWrapper(
            transformer=transformer, num_experts=4, expert_top_k=2,
            glu=True, mult=4, no_bias=True,
        )
        muon_opt, adamw_opt = configure_muon_optimizer(model, lr=1e-4, adamw_lr=1e-4)
        combo = MuonWithAdamW(muon_opt, adamw_opt)

        losses = []
        x = torch.randint(0, 100, (2, 16))
        for _ in range(5):
            loss = model(x)
            model.reset_moe_aux_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            combo.step()
            combo.zero_grad()
            losses.append(loss.item())

        assert len(losses) == 5
        assert all(not math.isnan(l) for l in losses)


import math


class TestEndToEnd:
    def test_moe_only_training(self):
        decoder = Decoder(dim=64, depth=2, heads=4, ff_glu=True, rotary_pos_emb=True)
        transformer = TransformerWrapper(num_tokens=100, max_seq_len=64, attn_layers=decoder)
        model = MoETransformerWrapper(
            transformer=transformer, num_experts=4, expert_top_k=2,
            glu=True, mult=4, no_bias=True,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, 100, (2, 16))
        initial_loss = model(x).item()
        for _ in range(10):
            loss = model(x)
            model.reset_moe_aux_loss()
            loss.backward()
            opt.step()
            opt.zero_grad()
        final_loss = model(x).item()
        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    def test_model_param_count(self):
        decoder = Decoder(dim=64, depth=2, heads=4, ff_glu=True, rotary_pos_emb=True)
        transformer = TransformerWrapper(num_tokens=100, max_seq_len=64, attn_layers=decoder)
        model = MoETransformerWrapper(
            transformer=transformer, num_experts=4, expert_top_k=2,
            glu=True, mult=4, no_bias=True,
        )
        assert model.num_params > 0
        assert model.num_trainable_params == model.num_params

    def test_replace_attention_standalone(self):
        from x_moe.attention import HybridAttentionBlock
        decoder = Decoder(dim=64, depth=2, heads=4, ff_glu=True, rotary_pos_emb=True)
        transformer = TransformerWrapper(num_tokens=100, max_seq_len=64, attn_layers=decoder)
        ds4_block = HybridAttentionBlock(
            dim=64,
            hca_config={"kv_dim": 32, "num_query_heads": 4, "compression_rate": 4, "window_size": 0},
        )
        model = MoETransformerWrapper(
            transformer=transformer, num_experts=4, expert_top_k=2,
            glu=True, mult=4, no_bias=True,
            ds4_attention=ds4_block,
        )
        ds4_count = sum(1 for m in model.modules() if isinstance(m, (DS4AttentionLayer, HybridAttentionBlock)))
        assert ds4_count >= 1