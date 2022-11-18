import os
import subprocess 
subdirs =  [f.path for f in os.scandir('.') if f.is_dir()]
crashed_count,synced_count  = 0,0
for path in subdirs :
    if 'offline' in path : 
        #cmd ='wandb sync ' + '~/Documents/GitHub/TeachmyAgent/wandb/'  "{}".format(str(path[2:])) +'/' 
        #subprocess.run([cmd],timeout=60)
        try : 
            print(path) 
            print("wandb sync {}".format(str(path[2:]))) 
            cmd = "wandb sync {}".format(str(path[2:])) 
            subprocess.run(cmd,timeout=30,shell=True)
            #os.system("wandb sync {}".format(str(path[2:]))) 
            synced_count +=1 
        except :
            crashed_count+=1 