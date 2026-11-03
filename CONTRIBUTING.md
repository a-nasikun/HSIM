# Contributing to HSIM

## Layout

| path | content |
|------|---------|
| `HSIM/include/hsim/` | public headers of the library (one module per header, `hsim.h` includes all) |
| `HSIM/src/` | implementation and private helpers (`logger.h`, `timer.h`) |
| `Example/` | applications built on the library (`1_HSIM_GUI`, `2_HSIM_CLI`) |
| `benchmarks/` | stage-timing program |
| `docs/` | description of the method and notation (`method.md`), Doxygen configuration |
| `tests/` | doctest test-suite, regression baselines in `tests/data/` |
| `cmake/` | dependency recipes and compiler-warning settings |

## Conventions

* C++17, namespace `hsim`; types in `PascalCase`, functions and variables in `camelCase`, private members with a
  trailing underscore. Mathematical names from the paper (`S`, `M`, `U`, `P`, `V`, `F`, `mu`, `lambda`) are kept.
* Public functions report invalid input with exceptions (`std::invalid_argument`, `std::out_of_range`,
  `std::logic_error` for calls in the wrong order, `std::runtime_error` for numerical failures). The library never
  prints unless asked to through `hsim::Options::verbosity`.
* Every tuning constant is a documented member of `hsim::Options`; no magic numbers in the algorithm code.
* Every public header must be self-contained (checked by the `hsim_header_check` target) and documented with Doxygen
  comments that state the mathematics, not only the code.
* Formatting is defined by `.clang-format`; static-analysis checks by `.clang-tidy`.

## Before submitting a change

```bash
clang-format -i HSIM/include/hsim/*.h HSIM/src/*.h HSIM/src/*.cpp tests/*.h tests/*.cpp Example/*/*.h Example/*/*.cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHSIM_WARNINGS_AS_ERRORS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

A change to the numerical method must keep the regression tests (`tests/test_regression.cpp`) passing. If results
change on purpose, regenerate the baselines and document the reason in `tests/data/README.md` and `CHANGELOG.md`.
