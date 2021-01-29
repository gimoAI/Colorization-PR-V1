
## PATTERN RECOGNITION: Image Colorization

### Training the model
The model first needs to be trained, as the checkpoint files for the different models are too big to be uploaded to GitHub.

### Dataset
- We use [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html). To train a model on the full dataset, download datasets from official websites.
After downloading, put then under the `datasets` folder in the main directory.

### Training:
- Open CMD
- type "cd ....\Colorization-COLAB-ready", this refers to the directory where the code is
- type "python train.py --seed 100 --dataset cifar10 --dataset-path ./dataset/cifar10 --checkpoints-path ./checkpoints --gpu-ids '0' --save-interval 50 --batch-size 128 --epochs 200 --lr 3e-4 --lr-decay-steps 1e4 --augment True"

### Testing:
Now it starts training the baseline model. If this process finishes, testing goes as follows:
- put desired test images in the "\checkpoints\test" folder
- Open CMD
- type "cd ....\Colorization-PR-V1", this refers to the directory where the code is
- type "python test.py --checkpoints-path ./checkpoints --test-input ./checkpoints/test --test-output ./checkpoints/output --cifar10"

To switch using to a different model:
- Copy the "src" folder from the desired model in the "Alternative models" and merge it into "Colorization-PR-V1"
- If you want to start training one of the other models, make sure to delete the "\checkpoints\cifar10" folder

### NOTE:
- We had to make many changes across the files to make the original code on github run (it is also based on TensorFlow 1)

## Citation
If you use this code for your research, please cite the paper <a href="https://arxiv.org/abs/1803.05400">Image Colorization with Generative Adversarial Networks</a>:

```
@inproceedings{nazeri2018image,
  title={Image Colorization Using Generative Adversarial Networks},
  author={Nazeri, Kamyar and Ng, Eric and Ebrahimi, Mehran},
  booktitle={International Conference on Articulated Motion and Deformable Objects},
  pages={85--94},
  year={2018},
  organization={Springer}
}
```
