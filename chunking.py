def chunk_hex_string(hex_str, chunk_size=32):
    return [hex_str[i:i+chunk_size] for i in range(0, len(hex_str), chunk_size)]

def process_embeddings_file(input_path="embeddings.txt", output_path="embeddings.txt"):
    with open(input_path, "r") as f:
        content = f.read().strip()

    # Step 1: Split by commas (preserve structure)
    parts = content.split(",")
    if len(parts) != 4:
        raise ValueError(f"Expected 4 comma-separated parts, got {len(parts)}")

    # Step 2: For each part, split into 8-char chunks
    processed_lines = []
    for i, part in enumerate(parts):
        part = part.strip().replace(" ", "").replace("\n", "")
        chunks = chunk_hex_string(part, chunk_size=32)
        for j, chunk in enumerate(chunks):
            line = chunk
            # Add comma after the last chunk of each part, except the last part
            if i < 3 and j == len(chunks) - 1:
                line += ","
            processed_lines.append(line)

    # Step 3: Write the result back to file
    with open(output_path, "w") as f:
        for line in processed_lines:
            f.write(line + "\n")

    print(f"Wrote {len(processed_lines)} lines to {output_path}")

if __name__ == "__main__":
    process_embeddings_file()
