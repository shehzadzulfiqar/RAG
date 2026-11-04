using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;

internal class Program
{
    static void Main()
    {
        #region paths
        string vocabPath = Path.Combine(
            AppContext.BaseDirectory,
            "Models",
            "tokenizer",
            "vocab.txt"
        );

        string modelPath = Path.Combine(
            AppContext.BaseDirectory,
            "Models",
            "model.onnx"
        );

        if (!File.Exists(vocabPath))
        {
            Console.WriteLine($"Vocabulary not found: {vocabPath}");
            return;
        }

        if (!File.Exists(modelPath))
        {
            Console.WriteLine($"Model not found: {modelPath}");
            return;
        }

        var tokenizer = BertTokenizer.Create(vocabPath);

        using var session = new InferenceSession(modelPath);

        // InspectOnnxModel(session);

        // ReadAndChunkDocument();


        string filePath = Path.Combine(
            AppContext.BaseDirectory,
            "Documents",
            "SampleText.txt"
        );

        if (!File.Exists(filePath))
        {
            Console.WriteLine($"Document not found: {filePath}");
            return;
        }

        string documentText = File.ReadAllText(filePath);

        // Split the document into sentence-based chunks.
        string[] chunks = documentText.Split(
            '.',
            StringSplitOptions.RemoveEmptyEntries |
            StringSplitOptions.TrimEntries
        );

        Console.WriteLine($"Total chunks: {chunks.Length}");
        #endregion
        var chunkEmbeddings = new List<float[]>();

        for (int i = 0; i < chunks.Length; i++)
        {
            string chunk = chunks[i] + ".";

            Console.WriteLine($"\nProcessing chunk {i + 1}: {chunk}");

            float[] embedding = GenerateEmbedding(
                chunk,
                tokenizer,
                session
            );

            chunkEmbeddings.Add(embedding);

            Console.WriteLine(
                $"Embedding generated: {embedding.Length} dimensions"
            );
        }

        // TASK 3: RETRIEVE THE MOST RELEVANT CHUNK FOR A QUESTION

        Console.WriteLine("\\nQUESTION RETRIEVAL");
        Console.WriteLine("------------------");

        string question = "provinces of Pakistan?";

        // Generate an embedding for the question.
        float[] questionEmbedding = GenerateEmbedding(
            question,
            tokenizer,
            session
        );

        double highestScore = double.MinValue;
        int bestChunkIndex = -1;

        // Compare the question embedding with every chunk embedding.
        for (int i = 0; i < chunkEmbeddings.Count; i++)
        {
            double score = CosineSimilarity(
                questionEmbedding,
                chunkEmbeddings[i]
            );

            Console.WriteLine(
                $"Chunk {i + 1} | Score: {score:F4} | {chunks[i]}"
            );

            if (score > highestScore)
            {
                highestScore = score;
                bestChunkIndex = i;
            }
        }

        // Display the best matching chunk.
        Console.WriteLine("\\nBEST MATCH");
        Console.WriteLine("----------");

        if (bestChunkIndex >= 0)
        {
            Console.WriteLine($"Question: {question}");
            Console.WriteLine(
                $"Relevant chunk: {chunks[bestChunkIndex]}"
            );
            Console.WriteLine($"Similarity score: {highestScore:F4}");
        }

    }


    // =========================================================
    // METHOD 1: READ AND CHUNK DOCUMENT
    // =========================================================

    static void ReadAndChunkDocument()
    {
        // Your original commented-out code is retained here.

        /*
        string filePath = Path.Combine(
            AppContext.BaseDirectory,
            "Documents",
            "SampleText.txt"
        );

        Console.WriteLine("Looking for file at:");
        Console.WriteLine(filePath);
        Console.WriteLine();

        if (!File.Exists(filePath))
        {
            Console.WriteLine("File not found.");
            return;
        }

        string text = File.ReadAllText(filePath);

        string[] chunks = text
            .Split(
                '.',
                StringSplitOptions.RemoveEmptyEntries |
                StringSplitOptions.TrimEntries
            );

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
        */
    }


    // =========================================================
    // METHOD 2: INSPECT ONNX MODEL INPUTS AND OUTPUTS
    // =========================================================

    static void InspectOnnxModel(InferenceSession session)
    {
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

        /*
        Your original model-inspection implementation used:

        using var session = new InferenceSession(modelPath);

        foreach (var input in session.InputMetadata)
        {
            // Print input name, type and dimensions.
        }

        foreach (var output in session.OutputMetadata)
        {
            // Print output name, type and dimensions.
        }

        The active implementation above performs the same inspection
        using the already-loaded session.
        */
    }


    // =========================================================
    // METHOD 3: GENERATE A SENTENCE EMBEDDING
    // =========================================================

    static float[] GenerateEmbedding(
        string text,
        BertTokenizer tokenizer,
        InferenceSession session)
    {
        Console.WriteLine("\n==================================");
        Console.WriteLine($"Original text: {text}");
        Console.WriteLine("==================================");

        // STEP 1: TOKENIZATION

        IReadOnlyList<int> tokenIds = tokenizer.EncodeToIds(text);

        Console.WriteLine($"Number of tokens: {tokenIds.Count}");

        Console.WriteLine(
            $"Token IDs: {string.Join(", ", tokenIds)}"
        );

        Console.WriteLine("\nSpecial token IDs:");

        Console.WriteLine(
            $"[CLS]: {tokenizer.ClassificationTokenId}"
        );

        Console.WriteLine(
            $"[SEP]: {tokenizer.SeparatorTokenId}"
        );

        Console.WriteLine(
            $"[PAD]: {tokenizer.PaddingTokenId}"
        );


        // STEP 2: PREPARE MODEL INPUT ARRAYS

        int tokenCount = tokenIds.Count;

        long[] inputIds = tokenIds
            .Select(id => (long)id)
            .ToArray();

        long[] attentionMask = Enumerable
            .Repeat(1L, tokenCount)
            .ToArray();

        long[] tokenTypeIds = new long[tokenCount];

        Console.WriteLine("\nModel inputs prepared:");

        Console.WriteLine(
            $"Input IDs count: {inputIds.Length}"
        );

        Console.WriteLine(
            $"Attention mask count: {attentionMask.Length}"
        );

        Console.WriteLine(
            $"Token type IDs count: {tokenTypeIds.Length}"
        );


        // STEP 3: CREATE TENSORS

        var inputIdsTensor = new DenseTensor<long>(
            new[] { 1, tokenCount }
        );

        var attentionMaskTensor = new DenseTensor<long>(
            new[] { 1, tokenCount }
        );

        var tokenTypeIdsTensor = new DenseTensor<long>(
            new[] { 1, tokenCount }
        );

        for (int i = 0; i < tokenCount; i++)
        {
            inputIdsTensor[0, i] = inputIds[i];

            attentionMaskTensor[0, i] =
                attentionMask[i];

            tokenTypeIdsTensor[0, i] =
                tokenTypeIds[i];
        }

        Console.WriteLine("\nTensors created successfully.");

        Console.WriteLine(
            $"Input tensor shape: [1, {tokenCount}]"
        );

        Console.WriteLine(
            $"Attention mask shape: [1, {tokenCount}]"
        );

        Console.WriteLine(
            $"Token type tensor shape: [1, {tokenCount}]"
        );


        // STEP 4: EXECUTE THE ONNX MODEL

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(
                "input_ids",
                inputIdsTensor
            ),

            NamedOnnxValue.CreateFromTensor(
                "attention_mask",
                attentionMaskTensor
            ),

            NamedOnnxValue.CreateFromTensor(
                "token_type_ids",
                tokenTypeIdsTensor
            )
        };

        using var results = session.Run(inputs);

        var result = results.FirstOrDefault(
            item => item.Name == "last_hidden_state"
        );

        if (result == null)
        {
            throw new InvalidOperationException(
                "The model did not return last_hidden_state."
            );
        }

        var outputTensor = result.AsTensor<float>();

        Console.WriteLine($"\nOutput name: {result.Name}");

        Console.WriteLine(
            $"Output dimensions: " +
            $"[{string.Join(", ", outputTensor.Dimensions.ToArray())}]"
        );

        Console.WriteLine(
            $"Total output values: {outputTensor.Length}"
        );


        // STEP 5: MEAN POOLING

        int actualTokenCount = outputTensor.Dimensions[1];

        int embeddingSize = outputTensor.Dimensions[2];

        float[] embedding = new float[embeddingSize];

        int validTokenCount = 0;

        for (int token = 0; token < actualTokenCount; token++)
        {
            // Ignore padding tokens.

            if (attentionMaskTensor[0, token] == 0)
            {
                continue;
            }

            validTokenCount++;

            for (int dimension = 0;
                 dimension < embeddingSize;
                 dimension++)
            {
                embedding[dimension] +=
                    outputTensor[0, token, dimension];
            }
        }

        if (validTokenCount == 0)
        {
            throw new InvalidOperationException(
                "No valid tokens were found."
            );
        }

        // Calculate the mean for each dimension.

        for (int dimension = 0;
             dimension < embeddingSize;
             dimension++)
        {
            embedding[dimension] /= validTokenCount;
        }

        Console.WriteLine(
            $"\nEmbedding dimensions: {embedding.Length}"
        );

        Console.WriteLine("First 10 embedding values:");

        for (int i = 0;
             i < Math.Min(10, embedding.Length);
             i++)
        {
            Console.WriteLine($"[{i}] = {embedding[i]}");
        }


        // STEP 6: L2 NORMALIZATION

        double squaredSum = 0;

        for (int i = 0; i < embedding.Length; i++)
        {
            squaredSum += embedding[i] * embedding[i];
        }

        double norm = Math.Sqrt(squaredSum);

        if (norm > 0)
        {
            for (int i = 0; i < embedding.Length; i++)
            {
                embedding[i] = (float)(embedding[i] / norm);
            }
        }

        double normalizedNorm = Math.Sqrt(
            embedding.Sum(
                value => (double)value * value
            )
        );

        Console.WriteLine(
            $"\nEmbedding dimensions: {embedding.Length}"
        );

        Console.WriteLine(
            $"L2 norm after normalization: {normalizedNorm:F6}"
        );

        Console.WriteLine("First 10 normalized values:");

        for (int i = 0;
             i < Math.Min(10, embedding.Length);
             i++)
        {
            Console.WriteLine(
                $"[{i}] = {embedding[i]:F6}"
            );
        }


        // STEP 7: RETURN THE EMBEDDING

        return embedding;
    }


    /// <summary>
    /// Calculates the cosine similarity between two vectors.
    /// </summary>
    /// <param name="vectorA"></param>
    /// <param name="vectorB"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
    static double CosineSimilarity(float[] vectorA, float[] vectorB)
    {
        if (vectorA.Length != vectorB.Length)
        {
            throw new ArgumentException(
                "Both embeddings must have the same dimensions.");
        }

        double dotProduct = 0;

        for (int i = 0; i < vectorA.Length; i++)
        {
            dotProduct += (double)vectorA[i] * vectorB[i];
        }

        return dotProduct;
    }

}