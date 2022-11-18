import json 
import jsonpickle 
import os 
if __name__ =="__main__" :
    import argparse 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--path', type=str , default = None , help = 'Specify the path to benchmark config file') 
    parser.add_argument('--env', type=str , default = 'ring' , help = 'Specify the env to run') 
    parser.add_argument('--array_id', type = int, default = 1, help = 'Enables running of a batch job')
    parser.add_argument('--res', type = str, default = None, help = 'Specify the results path')
    args = parser.parse_args() 
    assert args.path is not None 
    assert args.array_id is not None 
    with open(args.path+"/config.json","r") as jsonfile:
        configurations = dict(jsonpickle.decode(json.load(jsonfile))) 
        jsonfile.close() 

    config = configurations[str(args.array_id-1)]  
    if args.res is None : 
        config+= 'res=' +str(args.path) 
    else : 
        config += 'res=' +str(args.res)
    if args.env == 'ring' : 
        os.system('python automatic_vehicular_control/ring.py . ' + config)


