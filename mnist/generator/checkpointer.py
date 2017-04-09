import hashlib

def generate_checkpoint_path(base, model):
  model_hash = hashlib.sha1(model.to_json().encode('utf-8')).hexdigest()[0:8]
  return "%s.h5" % (base)
