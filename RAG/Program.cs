string filePath = Path.Combine(
    AppContext.BaseDirectory,
    "Documents",
    "SampleText.txt"
);

Console.WriteLine($"Looking for file at:");
Console.WriteLine(filePath);
Console.WriteLine();

if (!File.Exists(filePath))
{
    Console.WriteLine("File not found.");
    return;
}

string text = File.ReadAllText(filePath);

string[] chunks = text
    .Split('.', StringSplitOptions.RemoveEmptyEntries |
                StringSplitOptions.TrimEntries);

Console.WriteLine("Document split into chunks:");
Console.WriteLine();

for (int i = 0; i < chunks.Length; i++)
{
    Console.WriteLine($"Chunk {i + 1}:");
    Console.WriteLine(chunks[i] + ".");
    Console.WriteLine();
}
Console.WriteLine("Document Content:");
Console.WriteLine("------------------");
Console.WriteLine(text);






