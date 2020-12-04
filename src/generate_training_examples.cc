#include <glob.h>    // glob(), globfree()
#include <string.h>  // memset()

#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "src/proto/training_example.pb.h"

using namespace cv;
using namespace std::chrono;

std::mutex mtx;

std::string SerializeImage(const Mat& m) {
  std::vector<uchar> bytes;
  bool encode_ok = imencode(".jpg", m, bytes);
  if (!encode_ok) {
    std::cout << "Encode image failed!" << std::endl;
  }
  std::string res(bytes.begin(), bytes.end());
  return res;
}

std::vector<std::string> glob(const std::string& pattern) {
  using namespace std;

  // glob struct resides on the stack
  glob_t glob_result;
  memset(&glob_result, 0, sizeof(glob_result));

  // do the glob operation
  int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
  if (return_value != 0) {
    globfree(&glob_result);
    stringstream ss;
    ss << "glob() failed with return_value " << return_value << endl;
    throw std::runtime_error(ss.str());
  }

  // collect all the filenames into a std::list<std::string>
  vector<string> filenames;
  for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
    filenames.push_back(string(glob_result.gl_pathv[i]));
  }

  // cleanup
  globfree(&glob_result);

  // done
  return filenames;
}

class ExampleExtractor {
 public:
  ExampleExtractor(const std::string& video_file) {
    cap_.open(video_file);
    if (!cap_.isOpened()) {
      std::cout << "Open video file error: " << video_file << std::endl;
    }
    frames_.reserve(5);
    file_name_ = video_file;
  }

  super_resolution::TrainingExamples ExtractExamples(int max_num_examples) {
    super_resolution::TrainingExamples exps;
    while (frames_.size() < 5) {
      bool enqueue_res = EnqueueNextFrame();
      if (!enqueue_res) break;
    }
    if (frames_.size() < 5) {
      std::cout << "Not enough frames." << std::endl;
      return exps;
    }
    size_t frame_count = 0;

    while (exps.examples_size() < max_num_examples) {
      if (frame_count % 30 == 0) {
        auto training_ex = exps.add_examples();
        *training_ex = ExtractExample();
      }
      if (exps.examples_size() % 50 == 0 && frame_count % 30 == 0) {
        mtx.lock();
        std::cout << file_name_ << ": " << exps.examples_size()
                  << " examples..." << std::endl;
        mtx.unlock();
      }
      auto enqueue_res = EnqueueNextFrame();
      if (!enqueue_res) {
        std::cout << "Reached end of video." << std::endl;
        break;
      }
      frame_count++;
    }
    return exps;
  }

 private:
  bool EnqueueNextFrame() {
    if (frames_.size() >= 5) frames_.erase(frames_.begin());
    Mat f;
    bool read_frame_res = cap_.read(f);
    if (!read_frame_res) return false;
    frames_.push_back(f);
    return true;
  }

  super_resolution::TrainingExample ExtractExample() {
    super_resolution::TrainingExample training_ex;
    for (const auto& frame_mat : frames_) {
      Mat resized_frame;
      resize(frame_mat, resized_frame, Size(854, 480), 0.0, 0.0,
             cv::INTER_AREA);
      training_ex.add_frames(SerializeImage(resized_frame));
    }
    Mat resized_high_res_frame;
    resize(frames_.back(), resized_high_res_frame, Size(2562, 1440), 0.0, 0.0,
           cv::INTER_CUBIC);
    training_ex.set_high_res_frame(SerializeImage(resized_high_res_frame));
    return training_ex;
  }

  std::string file_name_;
  VideoCapture cap_;
  std::vector<Mat> frames_;
};

void ConvertVideoFile(const std::string& fn) {
  mtx.lock();
  std::cout << fn << std::endl;
  mtx.unlock();
  ExampleExtractor ee(fn);
  const auto exps = ee.ExtractExamples(2000);
  int64 timestamp = duration_cast<microseconds>(
                        high_resolution_clock::now().time_since_epoch())
                        .count();
  std::ostringstream outfile;
  outfile << timestamp << ".binarypb";
  mtx.lock();
  std::cout << "Saving to " << outfile.str() << std::endl;
  mtx.unlock();
  std::ofstream out(outfile.str());
  if (!out.is_open()) {
    std::cout << "Open file error." << std::endl;
  }
  out << exps.SerializeAsString();
  out.close();
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cout << "Wrong number of params." << std::endl;
    return -1;
  }

  auto video_files = glob(argv[1]);

  std::vector<std::thread> threads;

  for (int i = 0; i < video_files.size(); i++) {
    threads.push_back(std::thread(ConvertVideoFile, video_files[i]));
  }

  for (auto& th : threads) {
    th.join();
  }
  return 0;
}