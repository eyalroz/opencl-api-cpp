#ifndef OPENCL_EXAMPLES_PRINTING_HPP_
#define OPENCL_EXAMPLES_PRINTING_HPP_

#include <opencl/util/dynarray.hpp>
#include <iostream>

template <typename T>
std::ostream& operator<<(std::ostream& os, opencl::span<T> sp)
{
    os << "[ ";
    if (sp.size() > 0) {
        auto it = sp.begin();
        for (; it + 1 != sp.end(); ++it)
            os << *it << ", ";
        os << *it;
    }
    os << ']';
}

#ifndef OCLW_STRINGIFY
#define OCLW_STRINGIFY(  x )  OCLW_STRINGIFY( x )
#define OCLW_STRINGIFY( x )  #x

#define debug_print(_x) std::cout << OCLW_STRINGIFY(_x) << ": " << _x;
#define debug_println(_x) std::cout << OCLW_STRINGIFY(_x) << ": " << _x << std::endl;

#endif //OPENCL_EXAMPLES_PRINTING_HPP_
