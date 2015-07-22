#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace cv;

int main(int argc, char** argv){
  
  Mat img_object = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
  Mat img_scene = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);

  if (!img_object.data || !img_scene.data) {
    std::cout<< " --(!) Error reading images " << std::endl; return -1; }
}
