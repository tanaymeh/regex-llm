#include <pybind11/pybind11.h>
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <string>

namespace py = pybind11;

bool safe_check(std::string pattern, std::string subject, uint32_t match_limit) {
    int errornumber;
    PCRE2_SIZE erroroffset;

    pcre2_code *re = pcre2_compile(
        (PCRE2_SPTR)pattern.c_str(),
        PCRE2_ZERO_TERMINATED,
        0,
        &errornumber,
        &erroroffset,
        NULL
    );

    if (re == NULL) return false;

    pcre2_match_context *mcontext = pcre2_match_context_create(NULL);
    pcre2_set_match_limit(mcontext, match_limit);
    
    pcre2_match_data *match_data = pcre2_match_data_create_from_pattern(re, NULL);

    int rc = pcre2_match(
        re,
        (PCRE2_SPTR)subject.c_str(),
        subject.length(),
        0,
        0,
        match_data,
        mcontext
    );

    pcre2_match_data_free(match_data);
    pcre2_match_context_free(mcontext);
    pcre2_code_free(re);

    return rc >= 0;
}

bool safe_fullmatch(std::string pattern, std::string subject, uint32_t match_limit) {
    std::string wrapped_pattern = "^(?:" + pattern + ")$";
    
    return safe_check(wrapped_pattern, subject, match_limit);
}

PYBIND11_MODULE(fast_regex, m) {
    m.doc() = "Safe regex fullmatch via PCRE2";
    m.def("fullmatch", &safe_fullmatch, "Enforce fullmatch safely", 
          py::arg("pattern"), py::arg("subject"), py::arg("limit") = 100000);
}