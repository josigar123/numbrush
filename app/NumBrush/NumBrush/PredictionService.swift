import UIKit

struct Candidate: Decodable {
    let digit: Int
    let confidence: Float
}

struct PredictionResponse: Decodable {
    let predictions: [Candidate]
}

struct PredictionService {
    static let url = URL(string: "http://localhost:8000/predict")!

    static func predict(image: UIImage) async throws -> PredictionResponse {
        guard let imageData = image.pngData() else {
            throw URLError(.badURL)
        }

        let boundary = UUID().uuidString
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"digit.png\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/png\r\n\r\n".data(using: .utf8)!)
        body.append(imageData)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        request.httpBody = body

        let (data, _) = try await URLSession.shared.data(for: request)
        return try JSONDecoder().decode(PredictionResponse.self, from: data)
    }
}
