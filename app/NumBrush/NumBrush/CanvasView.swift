import SwiftUI

struct CanvasView: View {
    @State private var lines: [Line] = []
    @State private var currentLine: Line = Line()
    var onSubmit: (UIImage) -> Void
    var onClear: () -> Void

    var body: some View {
        VStack(spacing: 20) {
            Canvas { context, size in
                for line in lines + [currentLine] {
                    var path = Path()
                    path.addLines(line.points)
                    context.stroke(
                        path,
                        with: .color(.white),
                        style: StrokeStyle(lineWidth: 22, lineCap: .round, lineJoin: .round)
                    )
                }
            }
            .background(Color(white: 0.08))
            .clipShape(RoundedRectangle(cornerRadius: 20))
            .frame(width: 300, height: 300)
            .overlay(
                RoundedRectangle(cornerRadius: 20)
                    .strokeBorder(Color.white.opacity(0.08), lineWidth: 1)
            )
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        currentLine.points.append(value.location)
                    }
                    .onEnded { _ in
                        lines.append(currentLine)
                        currentLine = Line()
                    }
            )

            HStack(spacing: 12) {
                Button(action: {
                    lines = []
                    onClear()
                }) {
                    Text("Clear")
                        .font(.system(size: 16, weight: .medium))
                        .foregroundStyle(.white.opacity(0.7))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                        .background(Color.white.opacity(0.08))
                        .clipShape(RoundedRectangle(cornerRadius: 14))
                }

                Button(action: { onSubmit(snapshot()) }) {
                    Text("Predict")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(.black)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                        .background(Color.white)
                        .clipShape(RoundedRectangle(cornerRadius: 14))
                }
            }
            .frame(width: 300)
        }
    }

    func snapshot() -> UIImage {
        let renderer = ImageRenderer(
            content: Canvas { context, size in
                context.fill(
                    Path(CGRect(origin: .zero, size: size)),
                    with: .color(.black)
                )
                for line in lines {
                    var path = Path()
                    path.addLines(line.points)
                    context.stroke(
                        path,
                        with: .color(.white),
                        style: StrokeStyle(lineWidth: 22, lineCap: .round, lineJoin: .round)
                    )
                }
            }
            .frame(width: 300, height: 300)
        )
        return renderer.uiImage ?? UIImage()
    }
}

struct Line {
    var points: [CGPoint] = []
}
