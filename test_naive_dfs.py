import numpy as np
import time
from py_z3.classes import GameOfLifeGrid
from solvers import Z3PredecessorFinder, SATPredecessorFinder, ILPPredecessorFinder
from higher_order_solvers import NaiveDFSSolver
from generate_alphabet import GenerateText


def verify_solution(path: list) -> bool:
    """Verify that each grid in the path leads to the next one."""
    for i in range(len(path) - 1):
        g = GameOfLifeGrid(grid=path[i])
        g.compute_next()
        if not np.array_equal(g.next_grid, path[i + 1]):
            return False
    return True


def test_dfs_solver_simple():
    """Test the DFS solver with a simple pattern."""
    print("=" * 60)
    print("Test 1: Simple Pattern (3 steps back)")
    print("=" * 60)
    
    # Create a simple target grid
    target = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ])
    
    print("Target grid:")
    print(target)
    
    # Test with SAT solver
    finder = SATPredecessorFinder()
    dfs_solver = NaiveDFSSolver(finder, max_attempts_per_level=100, verbose=True)
    
    steps = 3
    start_time = time.time()
    solution = dfs_solver.solve(target, steps)
    elapsed = time.time() - start_time
    
    if solution:
        print(f"\n✓ Found solution in {elapsed:.2f}s")
        print(f"Solution path length: {len(solution)}")
        
        # Verify the solution
        if verify_solution(solution):
            print("✓ Solution verified!")
        else:
            print("✗ Solution verification failed!")
        
        # Print the path
        print("\nSolution path:")
        for i, grid in enumerate(solution):
            print(f"\nStep {i}:")
            print(grid)
        
        stats = dfs_solver.get_stats()
        print(f"\nStatistics:")
        print(f"  Total attempts: {stats['total_attempts']}")
        print(f"  Backtracks: {stats['backtracks']}")
        print(f"  Max depth reached: {stats['max_depth_reached']}")
    else:
        print(f"\n✗ No solution found in {elapsed:.2f}s")
        stats = dfs_solver.get_stats()
        print(f"Statistics:")
        print(f"  Total attempts: {stats['total_attempts']}")
        print(f"  Backtracks: {stats['backtracks']}")


def test_dfs_solver_alphabet():
    """Test the DFS solver with alphabet letters."""
    print("\n" + "=" * 60)
    print("Test 2: Alphabet Letters (varying steps)")
    print("=" * 60)
    
    gen = GenerateText()
    letters = "ABC"  # Start with a few letters
    
    solvers = [
        ("SAT", SATPredecessorFinder()),
        ("Z3", Z3PredecessorFinder()),
    ]
    
    for letter in letters:
        print(f"\n--- Letter: {letter} ---")
        target = gen.text_to_grid(letter)
        print(f"Grid size: {target.shape}")
        
        for solver_name, finder in solvers:
            print(f"\nSolver: {solver_name}")
            
            # Try different step counts
            for steps in [2, 3, 5]:
                print(f"  Steps: {steps}")
                
                dfs_solver = NaiveDFSSolver(finder, max_attempts_per_level=50, verbose=False)
                
                start_time = time.time()
                solution = dfs_solver.solve(target, steps)
                elapsed = time.time() - start_time
                
                if solution:
                    verified = verify_solution(solution)
                    stats = dfs_solver.get_stats()
                    print(f"    ✓ Found in {elapsed:.2f}s (attempts: {stats['total_attempts']}, backtracks: {stats['backtracks']})")
                    if not verified:
                        print(f"    ✗ Verification failed!")
                else:
                    stats = dfs_solver.get_stats()
                    print(f"    ✗ Failed in {elapsed:.2f}s (attempts: {stats['total_attempts']}, backtracks: {stats['backtracks']})")


def test_dfs_comparison():
    """Compare different solver backends for DFS."""
    print("\n" + "=" * 60)
    print("Test 3: Solver Backend Comparison")
    print("=" * 60)
    
    # Simple test grid
    target = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ])
    
    print("Target grid:")
    print(target)
    
    solvers = [
        ("SAT", SATPredecessorFinder()),
        ("Z3", Z3PredecessorFinder()),
        ("ILP", ILPPredecessorFinder()),
    ]
    
    steps = 4
    print(f"\nSearching {steps} steps back...")
    
    results = {}
    
    for solver_name, finder in solvers:
        print(f"\n{solver_name} Solver:")
        dfs_solver = NaiveDFSSolver(finder, max_attempts_per_level=100, verbose=False)
        
        start_time = time.time()
        solution = dfs_solver.solve(target, steps)
        elapsed = time.time() - start_time
        
        stats = dfs_solver.get_stats()
        results[solver_name] = {
            'success': solution is not None,
            'time': elapsed,
            'stats': stats,
            'verified': verify_solution(solution) if solution else False
        }
        
        if solution:
            print(f"  ✓ Success in {elapsed:.2f}s")
            print(f"    Attempts: {stats['total_attempts']}")
            print(f"    Backtracks: {stats['backtracks']}")
            print(f"    Verified: {results[solver_name]['verified']}")
        else:
            print(f"  ✗ Failed in {elapsed:.2f}s")
            print(f"    Attempts: {stats['total_attempts']}")
            print(f"    Backtracks: {stats['backtracks']}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"{'Solver':<10} {'Success':<10} {'Time (s)':<12} {'Attempts':<12} {'Backtracks':<12}")
    print("-" * 60)
    for solver_name, result in results.items():
        success = "✓" if result['success'] else "✗"
        print(f"{solver_name:<10} {success:<10} {result['time']:<12.2f} {result['stats']['total_attempts']:<12} {result['stats']['backtracks']:<12}")


if __name__ == "__main__":
    # Run all tests
    test_dfs_solver_simple()
    test_dfs_solver_alphabet()
    test_dfs_comparison()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
