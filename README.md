# Pytorch - Exemple d'exécution sur Calcul Canada
Documentations :
- [Grappe de calcul Béluga](https://docs.computecanada.ca/wiki/B%C3%A9luga)
- [Exécution des tâches](https://docs.computecanada.ca/wiki/Running_jobs/fr#T.C3.A2che_GPU_.28avec_processeur_graphique.29)
- [Exécution des tâches GPU](https://docs.computecanada.ca/wiki/Using_GPUs_with_Slurm/fr)
- [Transfert de données](https://docs.computecanada.ca/wiki/Transferring_data/fr)
- [Transfert de gros fichiers](https://docs.computecanada.ca/wiki/Globus/fr)


## 1. Téléchargement des *datasets* localement

a. Obtention du code
```console
mkdir ~/pytorch-exemple-calcul-canada
cd ~/pytorch-exemple-calcul-canada
git clone https://github.com/mamaheux/pytorch-exemple-calcul-canada.git code
```

b. Création d'un environnement virtuel et installation des dépendances
```console
cd ~/pytorch-exemple-calcul-canada
python3 -m venv venv
source venv/bin/activate
pip install -r code/requirements.txt
```

c. Téléchargement des *datasets*
```console
cd ~/pytorch-exemple-calcul-canada
source venv/bin/activate
cd code

python train.py --dataset_root ~/pytorch-exemple-calcul-canada/datasets \
  --output_path ~/pytorch-exemple-calcul-canada/output --dataset cifar10 \
  --model vanilla_cnn --learning_rate 0.001 --batch_size 128 --epoch_count 0

python train.py --dataset_root ~/pytorch-exemple-calcul-canada/datasets \
  --output_path ~/pytorch-exemple-calcul-canada/output --dataset cifar100 \
  --model vanilla_cnn --learning_rate 0.001 --batch_size 128 --epoch_count 0
   
python train.py --dataset_root ~/pytorch-exemple-calcul-canada/datasets \
  --output_path ~/pytorch-exemple-calcul-canada/output --dataset svhn \
  --model vanilla_cnn --learning_rate 0.001 --batch_size 128 --epoch_count 0

rm -r ~/pytorch-exemple-calcul-canada/datasets/cifar-10-batches-py
rm -r ~/pytorch-exemple-calcul-canada/datasets/cifar-100-python
```

## 2. Téléversement des *datasets* sur la grappe de calcul Béluga
```console
cd ~/pytorch-exemple-calcul-canada
ssh username@beluga.computecanada.ca "mkdir ~/pytorch-exemple-calcul-canada"
rsync --progress datasets/* username@beluga.computecanada.ca:~/pytorch-exemple-calcul-canada/datasets/
```

## 4. Copie du code sur la grappe de calcul Béluga
a. Connexion au nœud de connexion de la grappe de calcul Béluga
```console
ssh username@beluga.computecanada.ca
```

b. Obtention du code
```console
cd ~/pytorch-exemple-calcul-canada
git clone https://github.com/mamaheux/pytorch-exemple-calcul-canada.git code
```

## 5. Test du code en mode interactif
a. Connexion au nœud de connexion de la grappe de calcul Béluga
```console
ssh username@beluga.computecanada.ca
```

b. Demande d'un nœud de calcul ayant un GPU, 5 coeurs CPU et 16 Go de ram pour une durée de une heure
```console
salloc --gres=gpu:1 --cpus-per-task=5 --mem=16G --time=0-01:00
```

c. Changement du répertoire courant pour le répertoire temporaire du nœud de calcul
```console
cd $SLURM_TMPDIR/
```

d. Copie du code et des *datasets*
```console
cp -r ~/pytorch-exemple-calcul-canada .
```

e. Création d'un environnement virtuel et installation des dépendances
```console
cd $SLURM_TMPDIR/
module load python/3.6 cuda cudnn
virtualenv --no-download env
source env/bin/activate
pip install --no-index torch_gpu
pip install --no-index -r pytorch-exemple-calcul-canada/code/requirements.txt
```

f. Exécution de l'entraînement pour les trois *datasets*
```console
cd $SLURM_TMPDIR/pytorch-exemple-calcul-canada/code

python train.py --use_gpu --dataset_root $SLURM_TMPDIR/pytorch-exemple-calcul-canada/datasets \
  --output_path ~/pytorch-exemple-calcul-canada/output/cifar10 --dataset cifar10 \
  --model vanilla_cnn --learning_rate 0.001 --batch_size 128 --epoch_count 2

python train.py --use_gpu --dataset_root $SLURM_TMPDIR/pytorch-exemple-calcul-canada/datasets \
  --output_path ~/pytorch-exemple-calcul-canada/output/cifar100 --dataset cifar100 \
  --model vanilla_cnn --learning_rate 0.001 --batch_size 128 --epoch_count 2

python train.py --use_gpu --dataset_root $SLURM_TMPDIR/pytorch-exemple-calcul-canada/datasets \
  --output_path ~/pytorch-exemple-calcul-canada/output/svhn --dataset svhn \
  --model vanilla_cnn --learning_rate 0.001 --batch_size 128 --epoch_count 2
```

g. Fermeture de la session en mode interactif
```console
exit
```

10. Vérification de la présence des fichiers de sortie
```console
ls ~/pytorch-exemple-calcul-canada/output/cifar10
ls ~/pytorch-exemple-calcul-canada/output/cifar100
ls ~/pytorch-exemple-calcul-canada/output/svhn
```

## 6. Exécution d'entraînements en mode *batch*
a. Création d'un fichier *script* pour l'entraînement
```console
cd ~/pytorch-exemple-calcul-canada/
vim train.sh
```

Copiez le contenu suivant :
```console
#!/bin/bash
#SBATCH --gres=gpu:1              # Nombre de GPU
#SBATCH --cpus-per-task=5         # Nombre de CPU
#SBATCH --mem=16G                 # Quantité de mémoire
#SBATCH --time=0-01:00            # Durée (JJ-HH:MM)

# Copie du code et des datasets
cd $SLURM_TMPDIR/
cp -r ~/pytorch-exemple-calcul-canada .

# Création de l'environnement virtuel
cd $SLURM_TMPDIR/
module load python/3.6 cuda cudnn
virtualenv --no-download env
source env/bin/activate
pip install --no-index torch_gpu
pip install --no-index -r pytorch-exemple-calcul-canada/code/requirements.txt

# Exécution de l'entraînement
cd $SLURM_TMPDIR/pytorch-exemple-calcul-canada/code
python train.py --use_gpu --dataset_root $SLURM_TMPDIR/pytorch-exemple-calcul-canada/datasets \
  --learning_rate 0.001 --batch_size 128 --epoch_count 10 "$@"
```

b. Création d'un *script* pour lancer toutes les configurations d'entraînement
```console
cd ~/pytorch-exemple-calcul-canada/
vim start_training.sh
```

Copiez le contenu suivant :
```console
sbatch train.sh --output_path ~/pytorch-exemple-calcul-canada/output/cifar10/vanilla_cnn --dataset cifar10 --model vanilla_cnn
sbatch train.sh --output_path ~/pytorch-exemple-calcul-canada/output/cifar100/vanilla_cnn --dataset cifar10 --model vanilla_cnn
sbatch train.sh --output_path ~/pytorch-exemple-calcul-canada/output/svhn/vanilla_cnn --dataset svhn --model vanilla_cnn

sbatch train.sh --output_path ~/pytorch-exemple-calcul-canada/output/cifar10/dense_block_cnn --dataset cifar10 --model dense_block_cnn
sbatch train.sh --output_path ~/pytorch-exemple-calcul-canada/output/cifar100/dense_block_cnn --dataset cifar10 --model dense_block_cnn
sbatch train.sh --output_path ~/pytorch-exemple-calcul-canada/output/svhn/dense_block_cnn --dataset svhn --model dense_block_cnn
```

c. Lancement des entraînements
```console
sh ~/pytorch-exemple-calcul-canada/start_training.sh
```

Il est possible de surveiller l'exécution des entraînements à l'aide de :
```console
sq
```

d. Obtention de la sortie des entraînements sur l'ordinateur local
```console
exit
cd ~/pytorch-exemple-calcul-canada
rsync --progress -r username@beluga.computecanada.ca:~/pytorch-exemple-calcul-canada/output/ output/ 
```
