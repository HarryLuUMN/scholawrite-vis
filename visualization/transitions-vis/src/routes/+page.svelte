<script>
  import { goto } from '$app/navigation';

  function navigateToLabelTransition() {
    goto('/label-transition');
  }

  function navigateToIterationVisualization() {
    goto('/iteration-label-transition');
  }

  function navigateToTextComparison() {
    goto('/text-comparison');
  }
</script>

<style>
  main {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 80vh;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    gap: 24px;
  }

  h1 {
    font-size: 2.5rem;
    color: #333;
  }

  button {
    padding: 12px 28px;
    font-size: 1.1rem;
    background-color: #4e79a7;
    color: white;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    box-shadow: 0 4px 8px rgb(78 121 167 / 0.3);
    transition: background-color 0.25s ease;
  }

  button:hover {
    background-color: #3a5f8a;
  }
</style>

<main>
  <h1>Visualization Dashboard</h1>

  <button on:click={navigateToLabelTransition}>
    Go to Label Transition Visualization
  </button>

  <button on:click={navigateToIterationVisualization}>
    Go to Iteration Visualization
  </button>

  <button on:click={navigateToTextComparison}>
    Go to Text Comparison Visualization
  </button>
</main>
