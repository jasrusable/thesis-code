#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "my_app.h"

using namespace cv;
using namespace std;

template <class T>
T sum_test(T a, T b) {
	return a + b;
}

int main(int argc, char** argv) {
	cout << sum_test(5, 6);
	return 0;		
}
