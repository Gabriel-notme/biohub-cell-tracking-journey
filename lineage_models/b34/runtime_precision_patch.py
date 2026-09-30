"""Mixed-precision convolutions with FP32 TTA accumulation and association."""
def make_baseline_fast(source):
    anchor='predict_cmd = [sys.executable'
    assert source.count(anchor)==1
    injection='''# Tensor-core convolution inference; accumulation and tracking remain FP32.
os.environ['BIOHUB_REFERENCE_PRECISION']='fp16_convolution_fp32_accumulation_v1'
_precision_path=REPO_DIR/'scripts/predict_unet_transformer.py'
_precision_source=_precision_path.read_text()
_precision_old='    model.eval()\\n    return model, config["window_size"], downsample'
_precision_new='''+repr('''    model.eval()
    original_encode=model.encode
    def encode_mixed_precision(images):
        with torch.autocast('cuda',dtype=torch.float16):
            features,logits=original_encode(images)
        return features.float(),[x.float() for x in logits]
    model.encode=encode_mixed_precision
    return model, config["window_size"], downsample''')+'''
assert _precision_source.count(_precision_old)==1,'Mixed precision anchor mismatch'
_precision_path.write_text(_precision_source.replace(_precision_old,_precision_new))
print('REFERENCE_PRECISION: FP16 convolution, FP32 TTA/association',flush=True)
'''
    source=source.replace(anchor,injection+'\n'+anchor)
    source=source.replace('_inference_resume_env_keys = [',"_inference_resume_env_keys = ['BIOHUB_REFERENCE_PRECISION', ")
    anchor='            model.eval()\n            print(\'Loaded DeepCenter add-only gate checkpoint:\', checkpoint_path)'
    assert source.count(anchor)==1
    replacement='''            model.eval()
            original_forward=model.forward
            def mixed_precision_forward(x):
                with torch.autocast('cuda',dtype=torch.float16):
                    result=original_forward(x)
                return result.float()
            model.forward=mixed_precision_forward
            print('Loaded DeepCenter add-only gate checkpoint:', checkpoint_path)'''
    return source.replace(anchor,replacement)
