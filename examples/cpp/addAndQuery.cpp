#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/bruteforce.h"
#include <chrono>
#include <fstream>
#include <set>
#include <thread>
template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}

int main() {
    int dim = 256;               // Dimension of the elements
    int max_elements = 500000;     // Maximum number of elements, should be known beforehand
    int M = 256;                 // Tightly connected with internal dimensionality of the data
                                 // strongly affects the memory consumption
    int ef_construction = 200;   // Controls index search speed/build speed tradeoff
    int num_threads = 10; 

    // Initing index
    hnswlib::L2Space space(dim);
    std::string hnsw_path = "hnsw - 500000_256.bin";
    std::string brt_path = "brt_force - 500000_256.bin";
    hnswlib::HierarchicalNSW<float>* alg_hnsw ;
    hnswlib::BruteforceSearch<float>* brt_search ;
    alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);
    brt_search = new hnswlib::BruteforceSearch<float>(&space, brt_path);
    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::mt19937 rng1;
    rng1.seed(147);

    auto start_generate = std::chrono::high_resolution_clock::now();  // 记录开始时间

    std::cout << "begin to generate random data" << "\n";

    std::uniform_real_distribution<> distrib_real;
    float* data = new float[dim * max_elements];
    float* data_query = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
        data_query[i] = distrib_real(rng1);
    }

    auto end_generate = std::chrono::high_resolution_clock::now();  // 记录结束时间
    std::chrono::duration<double, std::milli> elapsed_generate = end_generate - start_generate;
    std::cout << "Time taken to generate random data: " << elapsed_generate.count() << " milliseconds\n";

    auto start_add_points = std::chrono::high_resolution_clock::now();  // 记录开始时间

    std::cout << "begin to add points to index" << "\n";
    
    // ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId) {
    //     alg_hnsw->addPoint((void*)(data + dim * row), row);
    //     brt_search->addPoint((void*)(data + dim * row), row);
    // });

    auto end_add_points = std::chrono::high_resolution_clock::now();  // 记录结束时间
    std::chrono::duration<double, std::milli> elapsed_add_points = end_add_points - start_add_points;
    std::cout << "Time taken to add points to index: " << elapsed_add_points.count() << " milliseconds\n";

    std::cout << "Already added points to index" << "\n";

    // std::string hnsw_path = "hnsw.bin";
    // std::string brt_path = "brt_force.bin";
    // alg_hnsw->loadIndex(hnsw_path);
    // brt_search->loadIndex(brt_path);

    // Query the elements and measure recall
    float correct = 0;
    auto start_query_total = std::chrono::high_resolution_clock::now();  // 记录总体开始时间
    float total_hnsw_time = 0.0;
    float total_brt_time = 0.0;

    std::ofstream outFile("output.txt");  // Open the file for writing

    for (int i = 1; i < 100; i++) {
        // Decide whether the point is for insertion or query
        bool is_insertion = (i % 1 != 0); // 10% for insertion, 90% for query

        if (is_insertion) {
            auto start_insert_hnsw = std::chrono::high_resolution_clock::now();  // 记录HNSW插入开始时间
            alg_hnsw->addPoint(data_query + i * dim, i);
            auto end_insert_hnsw = std::chrono::high_resolution_clock::now();  // 记录HNSW插入结束时间
            std::chrono::duration<double, std::milli> elapsed_insert_hnsw = end_insert_hnsw - start_insert_hnsw;

            auto start_insert_brt = std::chrono::high_resolution_clock::now();  // 记录bruteforce插入开始时间
            brt_search->addPoint(data_query + i * dim, i);
            auto end_insert_brt = std::chrono::high_resolution_clock::now();  // 记录bruteforce插入结束时间
            std::chrono::duration<double, std::milli> elapsed_insert_brt = end_insert_brt - start_insert_brt;

            outFile << "Insertion " << i << " time for HNSW: " << elapsed_insert_hnsw.count() << " milliseconds\n";
            outFile << "Insertion " << i << " time for Bruteforce: " << elapsed_insert_brt.count() << " milliseconds\n";
        } else {
            

            auto start_query_brt = std::chrono::high_resolution_clock::now();  // 记录bruteforce查询开始时间
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result_brt = brt_search->searchwithDistance(data_query + i * dim, 33);
            auto end_query_brt = std::chrono::high_resolution_clock::now();  // 记录bruteforce查询结束时间
            std::chrono::duration<double, std::milli> elapsed_brt_time = end_query_brt - start_query_brt;
            int result_num=result_brt.size();
            std::cout<<result_num<<std::endl;                                                                    //tiaozheng
            auto start_query_hnsw = std::chrono::high_resolution_clock::now();  // 记录HNSW查询开始时间
            // std::priority_queue<std::pair<float, hnswlib::labeltype>> result_hnsw = alg_hnsw->searchwithDistance(data_query + i * dim, 33);
             std::priority_queue<std::pair<float, hnswlib::labeltype>> result_hnsw = alg_hnsw->searchKnn(data_query + i * dim, result_num);
            auto end_query_hnsw = std::chrono::high_resolution_clock::now();  // 记录HNSW查询结束时间
            std::chrono::duration<double, std::milli> elapsed_hnsw_time = end_query_hnsw - start_query_hnsw;
            result_num=result_hnsw.size();
            std::cout<<result_num<<std::endl;
             std::cout<<std::endl; 
            outFile << "Query " << i << " time for HNSW: " << elapsed_hnsw_time.count() << " milliseconds\n";
            outFile << "Query " << i << " time for Bruteforce: " << elapsed_brt_time.count() << " milliseconds\n";
        }
    }

    auto end_query_total = std::chrono::high_resolution_clock::now();  // 记录总体结束时间
    std::chrono::duration<double, std::milli> elapsed_query_total = end_query_total - start_query_total;

    std::cout << "Total time taken for all operations: " << elapsed_query_total.count() << " milliseconds\n";

    // auto start_serialize = std::chrono::high_resolution_clock::now();  // 记录序列化开始时间

    // // Serialize index
   
    // // alg_hnsw->saveIndex(hnsw_path);
    // // brt_search->saveIndex(brt_path);
    // delete alg_hnsw;

    // auto end_serialize = std::chrono::high_resolution_clock::now();  // 记录序列化结束时间
    // std::chrono::duration<double, std::milli> elapsed_serialize = end_serialize - start_serialize;
    // std::cout << "Time taken to serialize index: " << elapsed_serialize.count() << " milliseconds\n";

    // auto start_deserialize = std::chrono::high_resolution_clock::now();  // 记录反序列化开始时间

    // // Deserialize index and check recall
    // alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);
    // correct = 0;
    // for (int i = 0; i < max_elements; i++) {
    //     auto start_query_deserialize = std::chrono::high_resolution_clock::now();  // 记录反序列化查询开始时间
    //     std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data_query + i * dim, 10);
    //     auto end_query_deserialize = std::chrono::high_resolution_clock::now();  // 记录反序列化查询结束时间
    //     std::chrono::duration<double, std::milli> elapsed_query_deserialize = end_query_deserialize - start_query_deserialize;

    //     while (!result.empty()) {
    //         result.pop();
    //     }

    //     hnswlib::labeltype label = result.top().second;
    //     if (label == i) correct++;

    //     while (!result.empty()) {
    //         result.pop();
    //     }

    // }

    // auto end_deserialize = std::chrono::high_resolution_clock::now();  // 记录反序列化结束时间
    // std::chrono::duration<double, std::milli> elapsed_deserialize = end_deserialize - start_deserialize;
    // std::cout << "Time taken to deserialize index: " << elapsed_deserialize.count() << " milliseconds\n";

    delete[] data;
    delete alg_hnsw;
    return 0;
}
