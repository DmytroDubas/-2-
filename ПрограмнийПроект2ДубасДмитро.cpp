#include <iostream>
#include <vector>
#include <numeric>
#include <execution>
#include <thread>
#include <chrono>
#include <random>
#include <iomanip>

using namespace std;

double compute_inner_product(const vector<double>& a, const vector<double>& b) {
    return inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

template <typename Policy>
double compute_transform_reduce(const vector<double>& a, const vector<double>& b, Policy policy) {
    return transform_reduce(policy, a.begin(), a.end(), b.begin(), 0.0);
}

double custom_parallel_inner_product(const vector<double>& a, const vector<double>& b, int K) {
    size_t n = a.size();
    vector<double> partial_sums(K, 0.0);
    vector<thread> threads;

    size_t chunk_size = n / K;
    for (int k = 0; k < K; ++k) {
        size_t start = k * chunk_size;
        size_t end = (k == K - 1) ? n : start + chunk_size;
        threads.emplace_back([&a, &b, start, end, k, &partial_sums]() {
            double temp = inner_product(a.begin() + start, a.begin() + end, b.begin() + start, 0.0);
            partial_sums[k] = temp;
            });
    }

    for (auto& th : threads) {
        th.join();
    }

    return accumulate(partial_sums.begin(), partial_sums.end(), 0.0);
}

void run_experiments(size_t N) {

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-1.0, 1.0);

    vector<double> a(N), b(N);
    generate(a.begin(), a.end(), [&]() { return dist(gen); });
    generate(b.begin(), b.end(), [&]() { return dist(gen); });

    cout << fixed << setprecision(2);
    cout << "\n=== Експерименти для N = " << N << " ===\n";

    auto start = chrono::high_resolution_clock::now();
    double result = compute_inner_product(a, b);
    auto end = chrono::high_resolution_clock::now();
    double time_no_policy = chrono::duration<double, milli>(end - start).count();
    cout << "Внутрішній добуток (без політики): " << time_no_policy << " мс\n";

    start = chrono::high_resolution_clock::now();
    result = compute_transform_reduce(a, b, execution::seq);
    end = chrono::high_resolution_clock::now();
    double time_seq = chrono::duration<double, milli>(end - start).count();
    cout << "transform_reduce (послідовний): " << time_seq << " мс\n";

    start = chrono::high_resolution_clock::now();
    result = compute_transform_reduce(a, b, execution::par);
    end = chrono::high_resolution_clock::now();
    double time_par = chrono::duration<double, milli>(end - start).count();
    cout << "transform_reduce (паралельний): " << time_par << " мс\n";

    start = chrono::high_resolution_clock::now();
    result = compute_transform_reduce(a, b, execution::par_unseq);
    end = chrono::high_resolution_clock::now();
    double time_par_unseq = chrono::duration<double, milli>(end - start).count();
    cout << "transform_reduce (паралельний неблокуючий): " << time_par_unseq << " мс\n";

    cout << "\nМій паралельний алгоритм:\n";
    cout << "K\tЧас (мс)\n";
    vector<pair<int, double>> k_times;
    double min_time = numeric_limits<double>::max();
    int best_k = 1;

    for (int K = 1; K <= 32; ++K) {
        start = chrono::high_resolution_clock::now();
        result = custom_parallel_inner_product(a, b, K);
        end = chrono::high_resolution_clock::now();
        double time_k = chrono::duration<double, milli>(end - start).count();
        cout << K << "\t" << time_k << "\n";
        k_times.emplace_back(K, time_k);

        if (time_k < min_time) {
            min_time = time_k;
            best_k = K;
        }
    }

    unsigned int hw_threads = thread::hardware_concurrency();
    cout << "\nНайкращий K: " << best_k << " (час: " << min_time << " мс)\n";
    cout << "Апаратна паралельність: " << hw_threads << "\n";
    cout << "Співвідношення (найкращий K / апаратна паралельність): " << static_cast<double>(best_k) / hw_threads << "\n";

    if (hw_threads < 32) {
        double sum_delta = 0.0;
        int count = 0;
        for (size_t i = hw_threads + 1; i < k_times.size(); ++i) {
            double delta_t = k_times[i].second - k_times[i - 1].second;
            if (delta_t > 0) {
                sum_delta += delta_t;
                count++;
            }
        }
        cout << "Закон масштабування: Для K > " << hw_threads;
        if (count == 0) {
            cout << ", немає значного зростання (можливий шум у вимірах).\n";
        }
        else {
            double avg_increase = sum_delta / count;
            cout << ", час зростає приблизно лінійно на середні "
                << avg_increase << " мс на додаткову нитку (через перемикання контексту та накладні витрати).\n";
        }
    }
    else {
        cout << "Закон масштабування: Апаратна паралельність >= 32, немає надлишкового K для аналізу.\n";
    }
}

int main() {
    // Compiler: Microsoft Visual C++ (MSVC) 19.44 (Visual Studio 2022)
    setlocale(LC_ALL, "ru");
    vector<size_t> sizes = { 1000000, 10000000, 100000000 };
    for (auto N : sizes) {
        run_experiments(N);
    }
    return 0;
}