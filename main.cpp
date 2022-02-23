#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "ticktock.h"
#include <tbb/tbb.h>

#define NOMINMAX
// TODO: 并行化所有这些 for 循环

template <class T, class Func>
std::vector<T> fill(std::vector<T> &arr, Func const &func)
{
    TICK(fill);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, arr.size()),
                      [&](tbb::blocked_range<size_t> r)
                      {
                          for (size_t i = r.begin(); i < r.end(); ++i)
                          {
                              arr[i] = func(i);
                          }
                      });
    TOCK(fill);
    return arr;
}

template <class T>
void saxpy(T a, std::vector<T> &x, std::vector<T> const &y)
{
    TICK(saxpy);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, x.size()),
                      [&](tbb::blocked_range<size_t> r)
                      {
                          for (size_t i = r.begin(); i < r.end(); ++i)
                          {
                              x[i] = a * x[i] + y[i];
                          }
                      });
    TOCK(saxpy);
}

template <class T>
T sqrtdot(std::vector<T> const &x, std::vector<T> const &y)
{
    TICK(sqrtdot);
    T ret = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, x.size()), 0,
        [&](tbb::blocked_range<size_t> r, T local_res)
        {
            for (size_t i = r.begin(); i < r.end(); ++i)
            {
                local_res += x[i] * y[i];
            }
            return local_res;
        },
        [](T x, T y)
        {
            return x + y;
        });
    ret = std::sqrt(ret);
    TOCK(sqrtdot);
    return ret;
}

template <class T>
T minvalue(std::vector<T> const &x)
{
    TICK(minvalue);

    T ret = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, x.size()), 0,
        [&](tbb::blocked_range<size_t> r, T local_res)
        {
            for (size_t i = r.begin(); i < r.end(); ++i)
            {
                if (x[i] < local_res)
                    local_res = x[i];
            }
            return local_res;
        },
        [](T x, T y)
        {
            return x < y ? x : y;
        });

    return ret;
}

//并行操作同一个容器，存在互斥关系
template <class T>
std::vector<T> magicfilter(std::vector<T> const &x, std::vector<T> const &y)
{
    TICK(magicfilter);
    std::mutex mtx;
    std::vector<T> res;
    int n = std::min(x.size(), y.size());
    //首先预留出数据的大小，防止多线程push_back造成效率低下
    res.reserve(n);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](tbb::blocked_range<size_t> r)
                      {
                          std::vector<T> local_a;
                          //对local_a预留一定的大小，该线程的数据先填补到这个局部数组中
                          local_a.reserve(r.size());
                          for (size_t i = r.begin(); i < r.end(); ++i)
                          {
                              if (x[i] > y[i])
                              {
                                  local_a.push_back(x[i]);
                              }
                              else if (y[i] > x[i] && y[i] > 0.5f)
                              {
                                  local_a.push_back(y[i]);
                                  local_a.push_back(x[i] * y[i]);
                              }
                          }
                          //该线程工作做完后再加锁拷贝到全局数组中
                          std::lock_guard grd(mtx);
                          std::copy(local_a.begin(), local_a.end(), std::back_insert_iterator(res));
                      });

    TOCK(magicfilter);
    return res;
}

template <class T>
T scanner(std::vector<T> &x)
{
    TICK(scanner);
    auto ret = tbb::parallel_scan(
        tbb::blocked_range<std::size_t>(0, x.size()), T{},
        [&](auto r, auto local_res, auto is_final)
        {
            for (auto i = r.begin(); i != r.end(); ++i)
            {
                local_res += x[i];
                if (is_final)
                {
                    x[i] = local_res;
                }
            }
            return local_res;
        },
        [](auto x, auto y)
        { return x + y; });

    TOCK(scanner);
    return ret;
}

int main()
{
    size_t n = 1 << 26;
    std::vector<float> x(n);
    std::vector<float> y(n);

    fill(x, [&](size_t i)
         { return std::sin(i); });
    fill(y, [&](size_t i)
         { return std::cos(i); });

    saxpy(0.5f, x, y);

    std::cout << sqrtdot(x, y) << std::endl;
    std::cout << minvalue(x) << std::endl;

    auto arr = magicfilter(x, y);
    std::cout << arr.size() << std::endl;

    scanner(x);
    std::cout << std::reduce(x.begin(), x.end()) << std::endl;

    return 0;
}
