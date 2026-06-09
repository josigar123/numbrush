import SwiftUI

struct ContentView: View {
    @State private var result: PredictionResponse? = nil
    @State private var isLoading = false

    var body: some View {
        ZStack {
            Color(white: 0.05).ignoresSafeArea()

            VStack(spacing: 32) {
                VStack(spacing: 4) {
                    Text("NumBrush")
                        .font(.system(size: 28, weight: .bold))
                        .foregroundStyle(.white)
                    Text("Draw a digit")
                        .font(.system(size: 14, weight: .regular))
                        .foregroundStyle(.white.opacity(0.4))
                }

                CanvasView(
                    onSubmit: { image in
                        Task {
                            isLoading = true
                            result = nil
                            result = try? await PredictionService.predict(image: image)
                            isLoading = false
                        }
                    },
                    onClear: {
                        result = nil
                    }
                )

                ZStack {
                    if isLoading {
                        ProgressView()
                            .tint(.white)
                            .transition(.opacity)
                    } else if let result {
                        HStack(spacing: 12) {
                            ForEach(Array(result.predictions.enumerated()), id: \.offset) { index, candidate in
                                PredictionCard(candidate: candidate, isPrimary: index == 0)
                            }
                        }
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                    }
                }
                .frame(height: 100)
                .animation(.spring(duration: 0.4), value: isLoading)
                .animation(.spring(duration: 0.4), value: result != nil)
            }
            .padding(.vertical, 48)
        }
    }
}

struct PredictionCard: View {
    let candidate: Candidate
    let isPrimary: Bool

    var body: some View {
        VStack(spacing: 6) {
            Text("\(candidate.digit)")
                .font(.system(size: isPrimary ? 48 : 36, weight: .bold, design: .rounded))
                .foregroundStyle(isPrimary ? .white : .white.opacity(0.5))

            Text("\(Int(candidate.confidence * 100))%")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(isPrimary ? .white.opacity(0.7) : .white.opacity(0.3))
        }
        .frame(width: isPrimary ? 120 : 90, height: 90)
        .background(isPrimary ? Color.white.opacity(0.1) : Color.white.opacity(0.04))
        .clipShape(RoundedRectangle(cornerRadius: 18))
        .overlay(
            RoundedRectangle(cornerRadius: 18)
                .strokeBorder(
                    isPrimary ? Color.white.opacity(0.2) : Color.white.opacity(0.06),
                    lineWidth: 1
                )
        )
    }
}

#Preview {
    ContentView()
}
