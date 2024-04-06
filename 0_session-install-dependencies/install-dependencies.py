# !pip install --upgrade pip
# !pip install --no-cache-dir --log 0_session-install-dependencies/pip-req.log -r 0_session-install-dependencies/requirements.txt

import subprocess

print(subprocess.run(["sh 0_session-install-dependencies/setup.sh"], shell=True))
