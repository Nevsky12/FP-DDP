#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  irk · expected.hpp
//
//  A minimal C++20 substitute for std::expected (C++23), covering exactly the
//  surface the integrator uses: construct from a value or from irk::unexpected,
//  contextual-bool test, operator* / operator->, and .error().  Backed by
//  std::variant so lifetimes of non-trivial alternatives (Solution, Failure)
//  are handled correctly.  Callers always gate access on the bool test, so the
//  accessors use std::get_if and never throw.
// ─────────────────────────────────────────────────────────────────────────────
#include <utility>
#include <variant>

namespace irk {

template <class E>
class unexpected {
public:
    explicit unexpected(E e) : err_(std::move(e)) {}
    const E&  error() const &  noexcept { return err_; }
    E&        error() &        noexcept { return err_; }
    E&&       error() &&       noexcept { return std::move(err_); }
private:
    E err_;
};
template <class E> unexpected(E) -> unexpected<E>;

template <class T, class E>
class expected {
public:
    expected(T v)             : v_(std::in_place_index<0>, std::move(v)) {}
    expected(unexpected<E> u) : v_(std::in_place_index<1>, std::move(u).error()) {}

    explicit operator bool() const noexcept { return v_.index() == 0; }
    bool     has_value()     const noexcept { return v_.index() == 0; }

    T&        operator*() &        noexcept { return *std::get_if<0>(&v_); }
    const T&  operator*() const &  noexcept { return *std::get_if<0>(&v_); }
    T&&       operator*() &&       noexcept { return std::move(*std::get_if<0>(&v_)); }
    T*        operator->()         noexcept { return  std::get_if<0>(&v_); }
    const T*  operator->() const   noexcept { return  std::get_if<0>(&v_); }

    E&        error() &        noexcept { return *std::get_if<1>(&v_); }
    const E&  error() const &  noexcept { return *std::get_if<1>(&v_); }
    E&&       error() &&       noexcept { return std::move(*std::get_if<1>(&v_)); }

private:
    std::variant<T, E> v_;
};

} // namespace irk
