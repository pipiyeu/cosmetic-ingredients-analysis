export async function analyzeIngredients(ingredients: string) {
  const response = await fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      ingredients,
    }),
  });

  if (!response.ok) {
    throw new Error("Failed to analyze ingredients");
  }

  return response.json();
}