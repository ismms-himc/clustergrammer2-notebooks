# these are two functions that save and load data in json

# save dict to json
def save_to_json(inst_dict, filename, indent=True):
  import json

  # save as a json
  fw = open(filename, 'w')
  if indent == True:
    fw.write( json.dumps(inst_dict, indent=2) )
  else:
    fw.write( json.dumps(inst_dict) )
  fw.close()


# load json to dict
def load_to_dict( filename ):
  import json
  # load
  f = open(filename,'r')
  inst_dict = json.load(f)
  f.close()
  return inst_dict
