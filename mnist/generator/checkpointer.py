import hashlib

def generate_checkpoint_path(base, model):
  model_hash = hashlib.sha1(model.to_json()).hexdigest()[0:8]
  return "%s_%s.h5" % (base, model_hash)
