#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

#define DIMENSION_UPPER_BOUND 200
#define TEMPLATE_LOWER_BOUND (DIMENSION_UPPER_BOUND * 0.1)
#define TEMPLATE_UPPER_BOUND (DIMENSION_UPPER_BOUND * 0.4)
char *tmplate_filename = "scumbag_hat.png";


void show_images (Mat image, Mat tmplate, Mat mask, Mat output) {
	imshow ("image", image);
	imshow ("tmplate", tmplate);
	imshow ("mask", mask);
	imshow ("output", output);
	int key = 0;
	while (key != 'q')
		key = waitKey (30);
}



/* Function: create_mask
 * ---------------------
 * given the template to make a mask for, this function will create the mask
 * and store it in mask
 */
void create_mask (Mat tmplate, Mat& mask) {
	/*### Step 4: get the mask for template matching ###*/
	mask = tmplate.clone ();
	cvtColor (mask, mask, CV_BGR2GRAY);
	threshold (mask, mask, 3, 1, THRESH_BINARY);
	cvtColor (mask, mask, CV_GRAY2BGR);
	assert ((mask.rows == tmplate.rows) && (mask.cols == tmplate.cols));
}


/* Function: get_scale_factor
 * --------------------------
 * given an image's dimensions, this function will scale it down so that the
 * larger dimension is no longer than DIMENSION_UPPER_BOUND
 */
float get_scale_factor (int image_width, int image_height, int larger_dimension) {

	float scale_factor;
	if (image_width > image_height)
		scale_factor = larger_dimension / float(image_width);
	else 
		scale_factor = larger_dimension / float(image_height);
	return scale_factor;
}


/* Function: scale_template_and_mask
 * ---------------------------------
 * given the original template and mask, then buffers to hold their resized versions,
 * this function will resize them appropriately
 */
void scale_template_and_mask (Mat tmplate, Mat& tmplate_resized, Mat mask, Mat& mask_resized, int larger_dimension) {

	assert ((tmplate.rows == mask.rows) && (tmplate.cols == mask.cols));
	float scale_factor = get_scale_factor (tmplate.cols, tmplate.rows, larger_dimension);
	resize (tmplate, tmplate_resized, 	Size(0, 0), scale_factor, scale_factor);
	resize (mask, mask_resized, 		Size(0, 0), scale_factor, scale_factor);
}


/* Function: calculate_match
 * -------------------------
 * given an image, the location to start at, and a template and mask, this function will calculate
 * the degree of match between that region in an image and the template (all areas on mask that aren't zero)
 */
double calculate_match (Mat image, Point start_coords, Mat tmplate, Mat mask) {

	Rect roi_rect = Rect(start_coords, Point(start_coords.x + tmplate.cols, start_coords.y + tmplate.rows));	
	Mat difference, final, temp;
	Mat masked_input = image(roi_rect).mul(mask);
	return norm (masked_input, tmplate, NORM_L2);	
}



/* Function: matchTemplateNonRectanguglar 
 * --------------------------------------
 * my own implementation of matchTemplate that takes into account 
 * a 'mask' - an image that has 0 marked at points that it shouldn't take
 * into account
 */
 void match_template_with_mask (Mat image, Mat tmplate, Mat mask, Mat& output) {

 	/*### Step 1: make sure channels are equal and that make sure the mask/template dimensions match ###*/
 	assert ((image.channels() == tmplate.channels()) && (image.channels() == mask.channels ()));
 	assert ((tmplate.rows == mask.rows) && (tmplate.cols == mask.cols));

	/*### Get dimensions of scan ###*/
	int scan_width = image.cols - tmplate.cols + 1;
	int scan_height = image.rows - tmplate.rows + 1;

	Point test_point = Point(1, 2);

	output = Mat (Size(scan_width, scan_height), CV_64F);
	uint8_t *output_start = output.data;

	for (int i=0;i<scan_height;i++) {
		for (int j=0;j<scan_width;j++) {
			/*--- perform scan at this location ---*/
			output.at<double>(i, j) = calculate_match (image, Point (j, i), tmplate, mask);
		}
	}

 }


int main( int argc, char** argv ) {

	/*### Step 0: set up displays for debugging ###*/
	namedWindow ("image", CV_WINDOW_AUTOSIZE);
	// namedWindow ("tmplate", CV_WINDOW_AUTOSIZE);
	// namedWindow ("mask", CV_WINDOW_AUTOSIZE);
	// namedWindow ("output", CV_WINDOW_AUTOSIZE);
	// namedWindow ("difference", CV_WINDOW_AUTOSIZE);

	/*### Step 1: load the hat template ###*/
	Mat tmplate;
	tmplate = imread (tmplate_filename, CV_LOAD_IMAGE_COLOR);
	// tmplate = Mat (Size(100, 100), CV_8UC3, Scalar(0, 0, 0));


	/*### Step 2: load the image to test ###*/
	Mat image;
	if (argc <= 1) {
		cout << "Error: specify an image file to search";
		return 0;
	}
	image = imread(argv[1], CV_LOAD_IMAGE_COLOR);


	/*### Step 3: downsample both so that they are pretty damn small ###*/
	float scale_factor = get_scale_factor (image.cols, image.rows, DIMENSION_UPPER_BOUND);
	resize (image, image, Size(0, 0), scale_factor, scale_factor);


	/*### Step 4: get the mask for template matching ###*/
	Mat mask;
	create_mask (tmplate, mask);


	/*### Step 5: iterate over all the possible sizes of the hat and to pattern matching ###*/
	double all_time_best = 10000;
	Rect best_rect;
	for (int i=TEMPLATE_LOWER_BOUND;i<TEMPLATE_UPPER_BOUND;i++) {

		/*--- resize template, mask ---*/
		Mat tmplate_resized, mask_resized;
		scale_template_and_mask (tmplate, tmplate_resized, mask, mask_resized, i);


		/*--- do template matching ---*/
		Mat output;
		match_template_with_mask (image, tmplate_resized, mask_resized, output);
		// matchTemplate (image, tmplate, output, CV_TM_SQDIFF);


		/* --- if the output is small, break the loop ---*/
		if ((output.rows <= 1) || (output.cols <= 1)) 
			break;


		/*--- find the best match locations in the image ---*/
		double min_val, max_val;
		Point min_loc, max_loc;
		minMaxLoc (output, &min_val, &max_val, &min_loc, &max_loc);


		int tmplate_pixels = tmplate_resized.cols * tmplate_resized.rows;
		min_val = min_val / tmplate_pixels;

		if (min_val < all_time_best) {
			all_time_best = min_val;
			best_rect = Rect(min_loc, Point(min_loc.x + tmplate_resized.cols, min_loc.y + tmplate_resized.rows));
		}


	}

	/*--- display info about the best fit ---*/
	Mat image_clone = image.clone ();
	rectangle(image_clone, best_rect, Scalar(255, 0, 0), 2, 8, 0);
	cout << "all time best: " << all_time_best << endl;
	imshow ("image", image_clone);
	int key = 0;
	while (key != 'q') 
		key = waitKey (30); 



	return 0;
}
 	