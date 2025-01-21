#ifndef OPENCL_WRAPPERS_IMPL_MISCELLANY_HPP_
#define OPENCL_WRAPPERS_IMPL_MISCELLANY_HPP_

#include <algorithm>

namespace opencl {
namespace detail {

template <typename Container>
auto get_handles(Container&& wrapped_objects) -> dynarray<decltype(std::begin(wrapped_objects)->handle())>
{
    using container_element_type = decltype(*std::begin(wrapped_objects));
    using handle_type = decltype(std::begin(wrapped_objects)->handle());
    auto get_handle = [](container_element_type obj) { return obj.handle(); };
    auto handles = make_dynarray<handle_type>(wrapped_objects.size());
    std::transform(std::begin(wrapped_objects), std::end(wrapped_objects), std::begin(handles), get_handle);
    return handles;
}

} // namespace detail
} // namespace opencl

#endif //OPENCL_WRAPPERS_IMPL_MISCELLANY_HPP_
