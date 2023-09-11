# NightTime Flare Removal

### Installation
1. Clone the repo
   ```
   git clone https://github.com/pknu-v-lab/NightTime_Flare_Removal
   ```

2. Install packages
   ```bash
   cd NIGHTTIME_FLARE_REMOVAL
   pip install -r requirements.txt
   ```






### Data Download

|     | Baidu Netdisk | Google Drive | Number | Description|
| :--- | :--: | :----: | :---- | ---- |
| Training Flare Images| [link](https://pan.baidu.com/s/1UlDPyZ_YRwyBpmPR1SXRwQ?pwd=mipi) | [link](https://drive.google.com/file/d/1eBSEayNuJqfwG-Md4PdeA_PJ3MQaqnXi/view?usp=share_link) | 5,000 |Scattering flares for train 
| Background Images| [link](https://pan.baidu.com/s/1BYPRCNSsVmn4VvuU4y4C-Q?pwd=zoyv) | [link](https://drive.google.com/file/d/1GNFGWfUbgXfELx5fZtjTjU2qqWnEa-Lr/view) | 23,949 | The background images are sampled from [[Single Image Reflection Removal with Perceptual Losses, Zhang et al., CVPR 2018]](https://people.eecs.berkeley.edu/~cecilia77/project-pages/reflection.html). We filter our most of the flare-corrupted images and overexposed images.|
| Validation images | [link](https://pan.baidu.com/share/init?surl=iNomlQuapPdJqtg3_uX_Fg&pwd=nips) | [link](https://drive.google.com/file/d/1PPXWxn7gYvqwHX301SuWmjI7IUUtqxab/view) | 100 | 100 validation images <br/> Flare7Kpp/test_data/real
| Test images | [link](https://pan.baidu.com/s/1fqvvxuCDMCwjLTORCJmG_g?pwd=mipi%20) | [link](https://drive.google.com/file/d/1-to2HVlgz-SD-xonXU1GrCN8n8d4B2T4/view?usp=share_link) | 100 | 100 Test images

### Structure
```
└──data
   ├── train 
   │    ├── flare
   │    └──Flickr24K 
   │
   ├── val
   └── test
```
