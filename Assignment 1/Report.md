# <u>**Video Stabilization** using **Lukas-Kanade Inverse Compositional Algorithm**</u>
> By **Anoop, 2015CS10265**

- The algorithm has been implemented using **Numpy** and **OpenCV** libraries in **Python3.5**.
- The implementation of Gauss-Newton process has been tested on some images and templates. It gives decent results in 500 iterations. This has been extended to video stabilization and has been tested on some videos. The testing videos had motion ranging from small tilts to extreme shakes. 
- The implementation handles motion in the testing videos to a great extent, keeping the main object still. The implementation could not handle the perspective motion present in the testing videos. Sometimes the Gauss-Newton process has been terminated before it could converge due to computational reasons. So few frames could be warped better.