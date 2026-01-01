#pragma once
#ifdef MINIFEM_USE_MPI
  #include <mpi.h>
#endif
#include <string>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <vector> 

namespace fem1d::mpi {

struct TimerDB {
  std::unordered_map<std::string, double> t; // seconds accumulated

  void add(const std::string& key, double dt) { t[key] += dt; }

  void print_rank0(MPI_Comm comm, int rank) const {
    // Reduce sums and max across ranks for each key (simple: print local on rank0 for now)
    if (rank == 0) {
      std::vector<std::pair<std::string,double>> items(t.begin(), t.end());
      std::sort(items.begin(), items.end(),
                [](auto& a, auto& b){ return a.second > b.second; });
      std::cout << "\n=== Timers (rank 0) ===\n";
      for (auto& kv : items) std::cout << kv.first << ": " << kv.second << " s\n";
    }
  }
};

struct ScopedTimer {
  TimerDB& db;
  std::string key;
  double t0;
  ScopedTimer(TimerDB& db_, std::string key_) : db(db_), key(std::move(key_)) {
#ifdef MINIFEM_USE_MPI
    t0 = MPI_Wtime();
#else
    t0 = 0.0;
#endif
  }
  ~ScopedTimer() {
#ifdef MINIFEM_USE_MPI
    db.add(key, MPI_Wtime() - t0);
#endif
  }
};

} // namespace fem1d::mpi
