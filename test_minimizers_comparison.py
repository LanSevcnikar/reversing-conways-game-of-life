#!/usr/bin/env python3
"""Comparison of all 6 solvers (Standard vs Minimizing) across the alphabet."""

import numpy as np
import time
import string
import sys
from py_z3.classes import GameOfLifeGrid
from solvers import Z3PredecessorFinder, SATPredecessorFinder, ILPPredecessorFinder
from minimizing_solvers import SATMinimizerSolver, Z3MinimizerSolver, ILPMinimizerSolver
from generate_alphabet import GenerateText

def verify_solution(predecessor: np.ndarray, target: np.ndarray) -> bool:
    game = GameOfLifeGrid(grid=predecessor)
    game.compute_next()
    return np.array_equal(game.next_grid, target)

def main():
    print("=" * 80)
    print("Comprehensive Solver Comparison (Standard vs Minimizing)")
    print("=" * 80)
    
    gen = GenerateText()
    letters = list(string.ascii_uppercase)
    # letters = ['A', 'B'] # For quick testing
    
    solvers = [
        ("Z3 Standard", Z3PredecessorFinder(), False),
        ("Z3 Minimizer", Z3MinimizerSolver(), True),
        ("SAT Standard", SATPredecessorFinder(), False),
        ("SAT Minimizer", SATMinimizerSolver(), True),
        ("ILP Standard", ILPPredecessorFinder(), False),
        ("ILP Minimizer", ILPMinimizerSolver(), True)
    ]
    
    # Store results: results[letter][solver_name] = (alive, time, verified)
    all_results = {}
    
    print(f"{'Letter':<6} | {'Solver':<15} | {'Alive':<6} | {'Time (s)':<8} | {'Verified':<8}")
    print("-" * 60)

    for letter in letters:
        try:
            target_grid = gen.text_to_grid(letter, padding=3)
        except Exception as e:
            print(f"Skipping {letter}: {e}")
            continue
            
        all_results[letter] = {}
        
        for name, solver, minimize_flag in solvers:
            start_time = time.time()
            try:
                # Run solver
                # Note: minimize=minimize_flag is passed to find_n_previous
                # Standard solvers might ignore it or it might be False for them
                solutions = solver.find_n_previous(target_grid, n=1, minimize=minimize_flag)
                
                elapsed = time.time() - start_time
                
                if solutions:
                    sol = solutions[0]
                    alive = np.sum(sol)
                    verified = verify_solution(sol, target_grid)
                    ver_str = "✓" if verified else "✗"
                    
                    all_results[letter][name] = (alive, elapsed, verified)
                    print(f"{letter:<6} | {name:<15} | {alive:<6} | {elapsed:<8.2f} | {ver_str:<8}")
                else:
                    all_results[letter][name] = (float('inf'), elapsed, False)
                    print(f"{letter:<6} | {name:<15} | {'N/A':<6} | {elapsed:<8.2f} | {'✗':<8}")
                    
            except Exception as e:
                elapsed = time.time() - start_time
                all_results[letter][name] = (float('inf'), elapsed, False)
                print(f"{letter:<6} | {name:<15} | {'Err':<6} | {elapsed:<8.2f} | {'Error':<8}")
                # print(f"  Error details: {e}")

    # Summary Analysis
    print("\n" + "=" * 80)
    print("Summary Analysis")
    print("=" * 80)
    
    # Compare Standard vs Minimizer for each type
    types = ["Z3", "SAT", "ILP"]
    
    for t in types:
        std_name = f"{t} Standard"
        min_name = f"{t} Minimizer"
        
        print(f"\nComparison: {std_name} vs {min_name}")
        print(f"{'Letter':<6} | {'Std Alive':<10} | {'Min Alive':<10} | {'Diff':<6} | {'Std Time':<10} | {'Min Time':<10}")
        print("-" * 70)
        
        better_count = 0
        total_count = 0
        
        for letter in letters:
            if letter not in all_results: continue
            
            if std_name in all_results[letter] and min_name in all_results[letter]:
                std_res = all_results[letter][std_name]
                min_res = all_results[letter][min_name]
                
                std_alive = std_res[0]
                min_alive = min_res[0]
                
                if std_alive == float('inf') or min_alive == float('inf'):
                    continue
                    
                diff = std_alive - min_alive
                if diff > 0: better_count += 1
                total_count += 1
                
                print(f"{letter:<6} | {std_alive:<10} | {min_alive:<10} | {diff:<6} | {std_res[1]:<10.2f} | {min_res[1]:<10.2f}")
        
        if total_count > 0:
            print(f"\nMinimizer found fewer alive cells in {better_count}/{total_count} cases.")

if __name__ == "__main__":
    main()
