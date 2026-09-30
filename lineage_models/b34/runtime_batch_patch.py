from pathlib import Path

def make_baseline_batched(source,fuse=False):
    helper=(Path(__file__).parent/'batched_d4.py').read_text()
    if fuse:helper+='\n'+(Path(__file__).parent/'fused_fp32.py').read_text()
    anchor='predict_cmd = [sys.executable'
    assert source.count(anchor)==1
    injection="""os.environ['BIOHUB_REFERENCE_PRECISION']='fp32_batched_d4_2_v1'
_batch_path=REPO_DIR/'scripts/predict_unet_transformer.py'
_batch_source=_batch_path.read_text()
_batch_old='    model.eval()\\n    return model, config["window_size"], downsample'
_batch_new='    model.eval()\\n    install_batched_d4(model,2)\\n    return model, config["window_size"], downsample'
assert _batch_source.count(_batch_old)==1,'Batched encode anchor mismatch'
assert float(os.environ['BIOHUB_SECONDARY_DETECTION_WEIGHT'])>0
_batch_source=_batch_source.replace(_batch_old,_batch_new)
_batch_source=_batch_source.replace('def load_model(',%r+'\\n\\ndef load_model(',1)
_batch_path.write_text(_batch_source)
print('REFERENCE_PRECISION: FP32 batched D4, original accumulation order',flush=True)
"""%helper
    if fuse:
        injection=injection.replace('fp32_batched_d4_2_v1','strict_fp32_fused_batched_d4_2_v1')
        injection=injection.replace('    install_batched_d4(model,2)',r'    install_fused_fp32(model)\n    install_batched_d4(model,2)')
    source=source.replace(anchor,injection+'\n'+anchor)
    return source.replace('_inference_resume_env_keys = [',"_inference_resume_env_keys = ['BIOHUB_REFERENCE_PRECISION', ")
