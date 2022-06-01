# Face relighting in Python
A python implementation of [portrait lighting transfer using a mass transport approach](https://www3.cs.stonybrook.edu/~cvl/content/papers/2017/shu_tog2017.pdf).\
3DMM model in original implementation is replaced by [PRNet](https://github.com/YadiraF/PRNet).

# Examples
input img | reference img | relighting with color | with light
![img](imgs/portrait_o1.jpg)
![img](imgs/portrait_o2.jpg)
![img](imgs/portrait_o3.jpg)
![img](imgs/portrait_o4.jpg)
![img](imgs/portrait_o5.jpg)
![img](imgs/portrait_o6.jpg)

# Environment
- python3.6
- dependencies 
    - download 256_256_resfcn256_weight.data-00000-of-00001 from [GoogleDrive](https://drive.google.com/file/d/1UoE-XuW1SDLUjZmJPkIZ1MLxvQFgmTFH/view?usp=sharing) and put into folder data/net-data/
    - run ```sh env.sh```
- Compile c++ files to .so for python use (ignore if you use numpy version) 
  ```bash
    cd python_portrait_relight/cython/
    python setup.py build_ext -i 
  ```

# Run
```bash
# python demo.py 
# fast = False(rendering with numpy version)
portrait_s1.jpg: 800x800x3
portrait_r1.jpg: 873x799x3
render_texture: 16.000
render_texture: 13.967
relight time with color: 36.16
render_texture: 13.871
render_texture: 13.617
relight time with light: 30.65
...

# python demo.py
# fast = True(rendering with c++ version)
portrait_s1.jpg: 800x800x3
portrait_r1.jpg: 873x799x3
render_texture: 0.026
render_texture: 0.024
relight time with color: 6.27
render_texture: 0.025
render_texture: 0.024
relight time with light: 3.20
...
```

# Methods

Let input image be I, reference image be R and output image be O.\
Let posI, posR be frontal 3d face position map of img I, R, with shape=[n, 3].\ 
Let colorI, colorR be rgb colors of the reconstructed vertices of img I, R, with shape=[n, 3].\
Let normalI, normalR be normal vectors of the vertices of img I, R, with shape=[n, 3].\
We obtain features fI=[colorI, posI[:,:,:2], nomralI], fR=[colorR, posR[:,:,:2], normalR] of img I, R, with shape=[n, 8].\
Then we determine pdf transfer function t, so that f{t(fI)}=f{fR}, where f{x} is the probability density function of array x.\
t(colorI) is the relighted image of I with R for reference.\
Finally, we apply regrain algorithm for postprocessing.

# Dependency
- 3D Face Reconstruction and Dense Alignment. [YadiraF/PRNet](https://github.com/YadiraF/PRNet.git)
- Face detection. [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface.git)
- pdf-transfer and regrain process. [pengbo-learn/python-color-transfer](https://github.com/pengbo-learn/python-color-transfer)


# References
[portrait lighting transfer using a mass transport approach](https://www3.cs.stonybrook.edu/~cvl/content/papers/2017/shu_tog2017.pdf) 
by Zhixin Shu, Sunil Hadap, Eli Shechtman, Kalyan Sunkavalli, Sylvain Paris and Dimitris Samaras.\
[Author's matlab implementation](https://github.com/AjayNandoriya/PortraitLightingTransferMTP)


