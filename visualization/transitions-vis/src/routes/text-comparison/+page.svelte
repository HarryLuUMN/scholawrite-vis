<script>
  import { goto } from '$app/navigation';

  // 示例文本，可替换为你的数据
  let beforeText = "This is the original version of the paragraph.";
  let afterText = "This is the revised and improved version of that paragraph.";

  function goBack() {
    goto('/');
  }
</script>

<style>
  .container {
    padding: 2rem;
    max-width: 900px;
    margin: 0 auto;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  }

  h2 {
    margin-bottom: 1rem;
    color: #333;
  }

  .text-box {
    border: 1px solid #ccc;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1.5rem;
    white-space: pre-wrap;
    background-color: #f9f9f9;
  }

  .buttons {
    display: flex;
    justify-content: flex-start;
    gap: 1rem;
  }

  button {
    padding: 10px 24px;
    background-color: #4e79a7;
    color: white;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 1rem;
    box-shadow: 0 4px 8px rgb(78 121 167 / 0.3);
    transition: background-color 0.25s ease;
  }

  button:hover {
    background-color: #3a5f8a;
  }
</style>

<div class="container">
  <h2>Text Comparison Visualization</h2>

  <div class="text-box">
    <strong>Before:</strong>
    <p>{beforeText}</p>
  </div>

  <div class="text-box">
    <strong>After:</strong>
    <p>{afterText}</p>
  </div>

  <div class="buttons">
    <button on:click={goBack}>← Back to Dashboard</button>
  </div>
</div>
