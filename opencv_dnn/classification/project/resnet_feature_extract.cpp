#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <filesystem>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace fs = std::filesystem;
using std::vector; using std::string; using std::cout; using std::cin;

// forward to inference engine
cv::Mat forward_output(
                                    cv::dnn::Net& model, //prediction model
                                    vector<cv::Mat>& images //images
                                    )
{
    //forward images through model to get output
    cv::Mat input_blobs = cv::dnn::blobFromImages(
        images=images,
        1.0, //scale factor
        cv::Size(224, 224), //size
        cv::Scalar(103.939, 116.779, 123.68),
        true, //swap RB
        false //center crop
    );

    //reshape blob to NHWC and set as input
    cv::Mat reshaped_blobs;
    cv::transposeND(input_blobs, vector<int>{0, 2, 3, 1}, reshaped_blobs);
    model.setInput(reshaped_blobs);

    /*
    * We will forward to the pooling layer, 126
    * and get (N, 2048) shape output
    * get the layer name
    */
    std::size_t layer_id = 126;
    vector<string> layers = model.getLayerNames();
    string layer_name = layers[layer_id];

    // forward
    cv::Mat output;
    output = model.forward(layer_name);
    return output;
}

int main(){
    // images to pass as model input at a go
    string model_path = "../../model/resnet50.onnx";
    string files_path = "images";

    // load the onnx model
    cv::dnn::Net resnet50 = cv::dnn::readNetFromONNX(model_path);
    if(resnet50.empty()){
        cout << "The model could not be found.!\n";
        return -1;
    }

    // read images
    if(files_path.empty()){
        cout << "The file path does not exist.!\n";
        return -1;
    }

    vector<string> extensions = {".png", ".jpg", ".jpeg", ".bmp"};
    fs::path searchpath(files_path); //path to find files
    vector<string> image_files;

    for(auto entry : fs::recursive_directory_iterator(searchpath)){
        //we only need files
        if(fs::is_directory(entry) == true){
            continue; //files only
        }
        else{
            string ext = entry.path().filename().extension().string();
            if(std::find(extensions.begin(), extensions.end(), ext) != extensions.end()){
                image_files.push_back(entry.path().string());
            }
        }
    }

    if(image_files.empty()){
        cout << "There are no images in the file.!\n";
        return -1;
    }

    cout << "Images: " << image_files.size() << "\n";

    cv::FileStorage ofile("data_final.xml", cv::FileStorage::WRITE);

    // We now have the images
    // We iterate through the list,
    // taking batch_size at a time.
    size_t batch_size = 50;
    size_t total_images = image_files.size();

    for(int take = 7000; take < total_images; take+=batch_size){
        vector<cv::Mat> images;
        // take batch_size images and forward
        int max_take = std::min(batch_size, total_images-take) + take;
        for(int path = take; path < max_take; path++){
            cv::Mat img, resized_img;
            img = cv::imread(image_files[path]);
            cv::resize(img, resized_img, cv::Size(256, 256));
            images.push_back(resized_img);
        }

        cv::Mat output = forward_output(resnet50, images);
        ofile << string("_") + std::to_string(take) << output;

        cout << "Done: " << take + batch_size << "\n";
    }
    ofile.release();

    return 0;
}
