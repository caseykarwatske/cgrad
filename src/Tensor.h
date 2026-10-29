#include <utility>
#include <functional>
#include <memory>
#include <vector>
#include <stdexcept>
#include <unordered_map>

template <typename T> 
struct TensorData {
  std::vector<T> data; 

  std::pair<std::shared_ptr<TensorData<T>>, std::shared_ptr<TensorData<T>>> children;

  std::vector<double> grad;
  std::function<void()> back;
};

template <typename T>
class Tensor {
  std::shared_ptr<TensorData<T>> data;

  Tensor(std::shared_ptr<TensorData<T>> data) {
    this->data = data;
  }

  public:
    Tensor(T data) {
      this->data = std::make_shared<TensorData<T>>();
      this->data->data = { data };
      this->data->grad.assign(1, 0.0);
    }

    T item() const {
      if (this->data->data.size() == 1)
        return this->data->data[0];
      throw std::invalid_argument("item() may not be called on Tensor of size greater than 1");
    }

    Tensor<T> operator+(const Tensor<T> &rhs) const {
      std::shared_ptr<TensorData<T>> new_tensor_data = std::make_shared<TensorData<T>>();

      new_tensor_data->data.resize(this->data->data.size());

      for (int i = 0; i < new_tensor_data->data.size(); i++)
        new_tensor_data->data[i] = this->data->data[i] + rhs.data->data[i];

      std::pair<std::shared_ptr<TensorData<T>>, std::shared_ptr<TensorData<T>>> children(this->data, rhs.data);
      new_tensor_data->children = children;

      new_tensor_data->grad.assign(this->data->data.size(), 0.0); 
      new_tensor_data->back = [lhs = this->data, rhs = rhs.data, res = std::weak_ptr<TensorData<T>>(new_tensor_data)]() -> void { // std::weak_ptr prevents circular dependency by not incrementing ref. count
        std::shared_ptr<TensorData<T>> resData = res.lock(); // allows dereferincing res by incrementing the ref. count during resData's lifetime to prevent destruction
        for (int i = 0; i < resData->data.size(); i++) {
          lhs->grad[i] += resData->grad[i]; 
          rhs->grad[i] += resData->grad[i];
        }
      };

      return Tensor<T>(new_tensor_data);
    }

    Tensor<T> operator*(const Tensor<T> &rhs) const {
      std::shared_ptr<TensorData<T>> new_tensor_data = std::make_shared<TensorData<T>>();

      new_tensor_data->data.resize(this->data->data.size());

      for (int i = 0; i < new_tensor_data->data.size(); i++)
        new_tensor_data->data[i] = this->data->data[i] * rhs.data->data[i];


      std::pair<std::shared_ptr<TensorData<T>>, std::shared_ptr<TensorData<T>>> children(this->data, rhs.data);
      new_tensor_data->children = children;

      new_tensor_data->grad.assign(this->data->data.size(), 0.0);
      new_tensor_data->back = [lhs = this->data, rhs = rhs.data, res = std::weak_ptr<TensorData<T>>(new_tensor_data)]() -> void {
        std::shared_ptr<TensorData<T>> resData = res.lock();
        for (int i = 0; i < resData->data.size(); i++) {
          lhs->grad[i] += rhs->data[i] * resData->grad[i];
          rhs->grad[i] += lhs->data[i] * resData->grad[i];
        }
      };

      return Tensor<T>(new_tensor_data);
    }

    void backward() {
      this->data->grad.assign(this->data->grad.size(), 1.0);

      // Kahn's topological sort
      std::vector<TensorData<T>*> nodes;

      std::unordered_map<TensorData<T>*, int> consumers;

      std::vector<TensorData<T>*> dfs;
      dfs.push_back(this->data.get());

      while (!dfs.empty()) {
        TensorData<T>* curr = dfs.back();
        dfs.pop_back();

        if (curr->children.first != nullptr) {
          if (consumers[curr->children.first.get()]++ == 0) {
            dfs.push_back(curr->children.first.get());
          }
        }

        if (curr->children.second != nullptr) {
          if (consumers[curr->children.second.get()]++ == 0) {
            dfs.push_back(curr->children.second.get());
          }
        }
      }

      nodes.push_back(this->data.get());

      while (!nodes.empty()) {
        TensorData<T>* curr = nodes.back();
        nodes.pop_back();

        if (curr->back)
          curr->back();

        if (curr->children.first != nullptr && --consumers[curr->children.first.get()] == 0)
          nodes.push_back(curr->children.first.get());
        if (curr->children.second != nullptr && --consumers[curr->children.second.get()] == 0)
          nodes.push_back(curr->children.second.get());
      }
    }
};
