package com.example.cagent.model

import java.util.UUID

data class ChatMessage(
    val id: String = UUID.randomUUID().toString(),
    val role: Role,
    val text: String,
)

enum class Role {
    User,
    Assistant,
    System,
}

