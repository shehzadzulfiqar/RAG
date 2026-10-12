//string filePath = Path.Combine(
//    AppContext.BaseDirectory,
//    "Documents",
//    "SampleText.txt"
//);

//Console.WriteLine($"Looking for file at:");
//Console.WriteLine(filePath);
//Console.WriteLine();

//if (!File.Exists(filePath))
//{
//    Console.WriteLine("File not found.");
//    return;
//}

//string text = File.ReadAllText(filePath);

//string[] chunks = text
//    .Split('.', StringSplitOptions.RemoveEmptyEntries |
//                StringSplitOptions.TrimEntries);

//Console.WriteLine("Document split into chunks:");
//Console.WriteLine();

//for (int i = 0; i < chunks.Length; i++)
//{
//    Console.WriteLine($"Chunk {i + 1}:");
//    Console.WriteLine(chunks[i] + ".");
//    Console.WriteLine();
//}
//Console.WriteLine("Document Content:");
//Console.WriteLine("------------------");
//Console.WriteLine(text);






///////////////////////////
///



using Microsoft.ML.OnnxRuntime;

string modelPath = Path.Combine(
    AppContext.BaseDirectory,
    "Models",
    "model.onnx"
);

if (!File.Exists(modelPath))
{
    Console.WriteLine($"Model not found: {modelPath}");
    return;
}

using var session = new InferenceSession(modelPath);

Console.WriteLine("MODEL INPUTS");
Console.WriteLine("============");

foreach (var input in session.InputMetadata)
{
    Console.WriteLine($"Name: {input.Key}");
    Console.WriteLine($"Type: {input.Value.ElementType}");
    Console.WriteLine(
        $"Dimensions: [{string.Join(", ", input.Value.Dimensions)}]"
    );
    Console.WriteLine();
}

Console.WriteLine("MODEL OUTPUTS");
Console.WriteLine("=============");

foreach (var output in session.OutputMetadata)
{
    Console.WriteLine($"Name: {output.Key}");
    Console.WriteLine($"Type: {output.Value.ElementType}");
    Console.WriteLine(
        $"Dimensions: [{string.Join(", ", output.Value.Dimensions)}]"
    );
    Console.WriteLine();
}
