import subprocess

while True:
    # rc = subprocess.run("sshpass -p '8452' rsync -avz --delete ./ basil@basil.local:./farm", shell=True)
    rc = subprocess.run("sshpass -p '8452' rsync -avz ./ basil@basil.local:/home/basil/", shell=True)

