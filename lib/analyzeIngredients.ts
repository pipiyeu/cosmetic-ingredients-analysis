export interface AnalyzeResult {
  ingredients: string;
  predictions: string[];
}

export async function analyzeIngredients(
  ingredients: string
): Promise<AnalyzeResult> {

  const response = await fetch(
    "/api/predict",
    {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify({
        ingredients,
      }),
    }
  );

  if (!response.ok) {

    let message = "Failed to analyze ingredients";

    try {

      const errorData = await response.json();

      if (errorData?.detail) {
        message = errorData.detail;
      }

    } catch {
      // Ignore JSON parsing error
    }

    throw new Error(message);
  }

  return response.json();
}