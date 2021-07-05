+#include <torch/torch.h>
#include <iostream>
#include <list>
#include "onnxruntime_cxx_api.h"

#define USE_CPU // Chnage 
#define Embed_size 512
int main() {


  //torch::Tensor tensor = torch::eye(3);
  //std::cout << tensor << std::endl;+

//torch::Tensor custom_group_norm(torch::Tensor input, int reps){
torch::Tensor repeatInterleave(torch::Tensor input , int reps){
    torch::Tensor red = torch::nn::AdaptiveAvgPoolImpl(input);
    //int reps = torch::Tensor::size(red , 1);
    red_repeat = at::Tensor repeat_interleave(red, reps));

    return red_repeat;

}

}

torch::Tensor reduction (
  torch::Tensor layerOne,
  torch::Tensor layerTwo,
  torch::Tensor layerThree,
  torch::Tensor layerFour) {
  torch ::Tensor one = torch::Tensor repeatInterleave(torch::Tensor layerOne , 8);
  torch ::Tensor two = torch::Tensor repeatInterleave(torch::Tensor layerTwo , 4);
  torch ::Tensor three = torch::Tensor repeatInterleave(torch::Tensor layerThree, 2);
  torch:: Tensor c = torch:: Tensor cat(torch ::Tensor one,torch ::Tensor two,
  torch ::Tensor three,torch ::Tensor layerFour);

  return torch::Tensor::mean(c);
 }

 ''' //std::vector<Ort::Value> inputTensors;
    //std::vector<Ort::Value> outputTensors;
   // Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
       // OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    //inputTensors.push_back(Ort::Value::CreateTensor<float>(
      ///  memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
       // inputDims.size()));
    ///outputTensors.push_back(Ort::Value::CreateTensor<float>(
       // memoryInfo, outputTensorValues.data(), outputTensorSize,
       // outputDims.data(), outputDims.size()));
 


static auto registry =
  torch::RegisterOperators("mynamespace::repeat_interleave", &repeat_interleave);

  
static auto registry2 =
  torch::RegisterOperators("mynamespace:reduction", &reduction);
