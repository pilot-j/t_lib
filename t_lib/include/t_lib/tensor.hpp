#include <iostream>
#include <vector>
#include <stdexcept>
#include <math.h>

namespace tensor_lib {

// Helper functions moved to namespace scope
int calculate_total(const std::vector<int>& shape) {
    if(shape.empty()) {
        throw std::logic_error("Shape vector should not be empty");
    }
    
    int total = shape[0];
    for(size_t i = 1; i < shape.size(); ++i) {
        total *= shape[i];
    }
    return total;
}

std::vector<int> calculate_stride(const std::vector<int>& shape, int total = 0) {
    if(shape.empty()) {
        throw std::logic_error("Shape vector should not be empty");
    }
    
    std::vector<int> stride;
    stride.reserve(shape.size()); // Pre-allocate memory
    
    if(total == 0) {
        total = calculate_total(shape);
    }
    
    for(const int& dim : shape) {
        stride.push_back(total / dim);
        total /= dim;
    }
    return stride;
}

template<typename T>
class Complex {
private:
    T real;
    T imag;
    T modulus;
    T phase;

public:
    Complex(T real = T{}, T imag = T{}) : real(real), imag(imag) {
        calculate_polar();
    }

    void calculate_polar() {
        modulus = std::sqrt(real * real + imag * imag);
        phase = std::atan2(imag, real);
    }

    // Getters
    T get_real() const { return real; }
    T get_imag() const { return imag; }
    T get_modulus() const { return modulus; }
    T get_phase() const { return phase; }
};

template<typename T>
class Tensor {
private:
    std::vector<int> shape;
    std::vector<int> stride;
    std::vector<T> elements;
    size_t total_elements;

public:
    Tensor() = default;
    
    Tensor(const std::vector<int>& shape, const std::vector<T>& elements = {}) {
        initialize(shape, elements);
    }

    void initialize(const std::vector<int>& shape, const std::vector<T>& elements = {});
    T at(const std::vector<int>& position, char print_mode = 'N') const;
    void print_dimensions() const;
    void print_tensor() const;
    
    // Element-wise operations
    template<typename Func>
    Tensor<T> element_wise_apply(Func operation) const;
    
    template<typename Func>
    Tensor<T> row_wise_apply(Func operation) const;
    
    template<typename Func>
    Tensor<T> column_wise_apply(Func operation) const;
    
    // Getters
    const std::vector<int>& get_shape() const { return shape; }
    const std::vector<T>& get_elements() const { return elements; }
    size_t get_total_elements() const { return total_elements; }
};

// Implementation of Tensor member functions
template<typename T>
void Tensor<T>::initialize(const std::vector<int>& input_shape, const std::vector<T>& input_elements) {
    shape = input_shape;
    total_elements = calculate_total(shape);
    stride = calculate_stride(shape, total_elements);
    
    if(input_elements.empty()) {
        elements.resize(total_elements);
    } else if(input_elements.size() != total_elements) {
        throw std::logic_error("Input data size does not match tensor shape");
    } else {
        elements = input_elements;
    }
}

template<typename T>
T Tensor<T>::at(const std::vector<int>& position, char print_mode) const {
    if(position.size() != shape.size()) {
        throw std::invalid_argument("Position input should match shape dimensions");
    }
    
    size_t flat_index = 0;
    for(size_t i = 0; i < shape.size(); ++i) {
        if(position[i] >= shape[i]) {
            throw std::out_of_range("Position index out of bounds");
        }
        flat_index += position[i] * stride[i];
    }
    
    if(print_mode == 'Y') {
        std::cout << "Element at given position: " << elements[flat_index] << '\n';
    }
    return elements[flat_index];
}

template<typename T>
template<typename Func>
Tensor<T> Tensor<T>::element_wise_apply(Func operation) const {
    std::vector<T> new_elements;
    new_elements.reserve(elements.size());
    
    for(const T& element : elements) {
        new_elements.push_back(operation(element));
    }
    
    return Tensor<T>(shape, new_elements);
}

} // namespace tensor_lib