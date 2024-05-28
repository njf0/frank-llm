from huggingface_hub import HfApi, HfFolder

hf_access_token = "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"

HfFolder.save_token(hf_access_token)
api = HfApi()
user = api.whoami(hf_access_token)
print(user)