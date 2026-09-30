"""Keep the dominant detector, its TTA and the DeepCenter gate in FP32."""
def make_baseline_selective(source):
    anchor='predict_cmd = [sys.executable';assert source.count(anchor)==1
    injection="""os.environ['BIOHUB_REFERENCE_PRECISION']='primary_fp16_secondary_fp32_full_d4_v1'
_selective_path=REPO_DIR/'scripts/predict_unet_transformer.py'
_selective_source=_selective_path.read_text()
_selective_old='    model, window_size, downsample = load_model(weights_path, device)'
_selective_new="""+repr('''    model, window_size, downsample = load_model(weights_path, device)
    original_primary_encode=model.encode
    def primary_encode_mixed(images):
        with torch.autocast('cuda',dtype=torch.float16):
            features,logits=original_primary_encode(images)
        return features.float(),[x.float() for x in logits]
    model.encode=primary_encode_mixed''')+"""
assert _selective_source.count(_selective_old)==1,'Selective precision anchor mismatch'
_selective_path.write_text(_selective_source.replace(_selective_old,_selective_new))
print('REFERENCE_PRECISION: primary FP16, dominant secondary and DeepCenter FP32; all D4 views retained',flush=True)
"""
    source=source.replace(anchor,injection+'\n'+anchor)
    return source.replace('_inference_resume_env_keys = [',"_inference_resume_env_keys = ['BIOHUB_REFERENCE_PRECISION', ")
