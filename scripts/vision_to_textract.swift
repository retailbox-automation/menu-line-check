// Apple Vision OCR (local, $0) written in the shape of an Amazon Textract DetectDocumentText
// response, so menu_line_check.ocr.parse_textract / parse_sheet read it unchanged. Only LINE blocks, top-left origin.
// Build: swiftc -O vision_to_textract.swift -o vision_to_textract   Run: ./vision_to_textract photo.jpg > out.json
import AppKit
import Foundation
import Vision

guard CommandLine.arguments.count == 2 else {
    FileHandle.standardError.write("usage: vision_to_textract <image>\n".data(using: .utf8)!)
    exit(2)
}
let path = CommandLine.arguments[1]
guard let img = NSImage(contentsOfFile: path),
      let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    FileHandle.standardError.write("cannot open image \(path)\n".data(using: .utf8)!)
    exit(1)
}
let req = VNRecognizeTextRequest()
req.recognitionLevel = .accurate
req.usesLanguageCorrection = false
do {
    try VNImageRequestHandler(cgImage: cg, options: [:]).perform([req])
} catch {
    FileHandle.standardError.write("vision failed: \(error)\n".data(using: .utf8)!)
    exit(1)
}
var blocks: [[String: Any]] = []
for (i, o) in (req.results ?? []).enumerated() {
    guard let c = o.topCandidates(1).first else { continue }
    let b = o.boundingBox  // normalised, origin bottom-left
    blocks.append([
        "BlockType": "LINE",
        "Id": "vision-\(i)",
        "Text": c.string,
        "Confidence": Double(c.confidence) * 100.0,
        "Geometry": ["BoundingBox": [
            "Left": Double(b.minX), "Top": Double(1.0 - b.maxY),
            "Width": Double(b.width), "Height": Double(b.height),
        ]],
    ])
}
let doc: [String: Any] = [
    "DocumentMetadata": ["Pages": 1],
    "Blocks": blocks,
    "_source": "Apple Vision VNRecognizeTextRequest (accurate), local; Textract-shaped",
]
let data = try JSONSerialization.data(withJSONObject: doc, options: [.prettyPrinted, .sortedKeys])
FileHandle.standardOutput.write(data)
FileHandle.standardOutput.write("\n".data(using: .utf8)!)
