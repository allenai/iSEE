# <a href="https://openaccess.thecvf.com/content/CVPR2022/papers/Dwivedi_What_Do_Navigation_Agents_Learn_About_Their_Environment_CVPR_2022_paper.pdf">What do navigation agents learn about their environment?</a>
#### Kshitij Dwivedi, Gemma Roig, Aniruddh Kembhavi, Roozbeh Mottaghi
#### CVPR 2022
#### <a href="https://kshitijd20.github.io/navigation_interpretability/">Project Page</a> 

We present <b>iSEE</b>, a framework that allows interpreting the dynamic representation of the navigation agents in terms of human interpretable concepts. 
The navigation agents were trained using <a href=https://allenact.org/>AllenAct</a> framework. In this repository, we provide:
<ul>
<li> Dataset of RNN activations and concepts
<li> Code to evaluate how well trained agents predict concepts
<li> Code to get top-K relevant neurons for predicting a given concept
</ul>

### Citation

If you find this project useful in your research, please consider citing:

```
   @InProceedings{dwivedi2022navigation_interpretability,
    title = {What do navigation agents learn about their environment?},
    author = {Kshitij Dwivedi, Gemma Roig, Aniruddh Kembhavi, Roozbeh Mottaghi},
    booktitle = {CVPR},
    year = {2022},
    }
```

### Contents
<div class="toc">
<ul>
<li><a href="#-installation">💻 Installation</a></li>
<li><a href="#-dataset">📊 Dataset</a></li>
<li><a href="#concept-prediction">Concept Prediction</a></li>
<li><a href="#topk-neurons">Top-k neurons</a></li>
</ul>
</div>

## 💻 Installation
 
To begin, clone this repository locally
```bash
git clone https://github.com/allenai/iSEE.git
```
Install anaconda and create a new conda environment
```bash
conda create -n iSEE
conda activate iSEE
```
Install xgboost-gpu using the following command
```bash
conda install -c anaconda py-xgboost-gpu
```
Install other requirements
```bash
pip install -r requirements.txt
```
## 📊 Dataset

Please download the dataset from <a href="https://kshitijd20.github.io/navigation_interpretability/">here</a>. Then unzip it inside data directory

## Concept Prediction
Run the following script to evaluate how well concepts can be predicted by trained agent (Resnet-objectnav) and compare it to corresponding untrained baseline
```bash
python predict concepts.py --model resnet --task objectnav
```
<details>
<summary>Arguments:</summary>

+ ```--model```: We used two architectures Resnet and SimpleConv. Options are ```resnet``` and ```simpleconv```
+ ````--task````: Options are ```objectnav``` and ```pointnav```
</details>

The script will generate plots and save them in ```results/task/model/plots``` directory

## TopK neurons
Run the following script to find which neurons were most relevant in predicting a given concept (e.g. front reachability) by a trained agent (Resnet-objectnav). 
```bash
python get_topk_neurons.py --model resnet --task objectnav --concept reachable_R=2_theta=000
```

<details>
<summary>Arguments:</summary>

+ ```--model```: We used two architectures Resnet and SimpleConv. Options are ```resnet``` and ```simpleconv```
+ ````--task````: Options are ```objectnav``` and ```pointnav```
+ ````--concept````: The concepts used in the paper are ```reachable_R=2_theta=000``` (Reachability at 2xgridSize and front) and ```target_visibility```. For full list of concepts in the dataset, please
refer to columns of ```data/trajectory_dataset/train/objectnav_ithor_default_resnet_pretrained/metadata.pkl``` file. 
</details>

The script will generate SHAP beeswarm plot for the concept and save it in ```results/task/model/shap_plots``` directory.

## Acknowledgements
We thank the <a href = https://github.com/slundberg/shap>SHAP</a> authors for easy to use code and <a href = https://github.com/allenai/manipulathor>ManipulaThor</a> authors for Readme template.
