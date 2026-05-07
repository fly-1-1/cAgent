package com.example.cagent.ui

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.cagent.model.Role
@Composable
fun ChatScreen(
    onOpenModels: () -> Unit,
    vm: ChatViewModel = viewModel(),
) {
    val state by vm.state.collectAsState()
    val listState = rememberLazyListState()

    LaunchedEffect(Unit) {
        vm.loadDefaultModelIfPresent()
    }

    // Auto-scroll to the latest message when the list grows or the last
    // message's text grows (streaming token append).
    val lastIndex = state.messages.lastIndex.coerceAtLeast(0)
    val lastLen = state.messages.lastOrNull()?.text?.length ?: 0
    LaunchedEffect(lastIndex, lastLen) {
        if (state.messages.isNotEmpty()) {
            listState.animateScrollToItem(lastIndex)
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = onOpenModels) { Text("Models") }
            OutlinedButton(onClick = { vm.newChat() }) { Text("New chat") }
        }

        Text(state.status, style = MaterialTheme.typography.bodySmall)

        LazyColumn(
            state = listState,
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            items(state.messages, key = { it.id }) { msg ->
                ChatMessageBubble(role = msg.role, text = msg.text)
            }
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            TextField(
                modifier = Modifier.weight(1f),
                value = state.draft,
                onValueChange = vm::updateDraft,
                placeholder = { Text("Type message…") },
                singleLine = true,
            )
            Button(
                onClick = { vm.send() },
                enabled = state.canSend,
            ) {
                Text("Send")
            }
        }
    }
}

@Composable
private fun ChatMessageBubble(
    role: Role,
    text: String,
) {
    val prefix = when (role) {
        Role.User -> "You"
        Role.Assistant -> "Assistant"
        Role.System -> "System"
    }
    val body = if (role == Role.Assistant && text.isEmpty()) "…" else text
    val parts = rememberMessageParts(body)

    Card(
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(prefix, style = MaterialTheme.typography.labelMedium)
            parts.forEach { p ->
                when (p) {
                    is MessagePart.Text -> SelectionContainer {
                        Text(p.value, style = MaterialTheme.typography.bodyMedium)
                    }
                    is MessagePart.Code -> CodeBlock(p.value)
                }
            }
        }
    }
}

@Composable
private fun CodeBlock(code: String) {
    val clipboard = LocalClipboardManager.current
    val scroll = rememberScrollState()

    Surface(
        tonalElevation = 2.dp,
        shape = MaterialTheme.shapes.medium,
        modifier = Modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.surfaceContainerHighest,
    ) {
        Column(modifier = Modifier.padding(10.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Text("Code", style = MaterialTheme.typography.labelSmall)
                OutlinedButton(onClick = { clipboard.setText(AnnotatedString(code)) }) {
                    Text("Copy")
                }
            }
            Spacer(Modifier.height(6.dp))
            SelectionContainer {
                Box(modifier = Modifier.horizontalScroll(scroll)) {
                    Text(
                        text = code,
                        style = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace),
                    )
                }
            }
        }
    }
}

private sealed interface MessagePart {
    data class Text(val value: String) : MessagePart
    data class Code(val value: String) : MessagePart
}

@Composable
private fun rememberMessageParts(text: String): List<MessagePart> {
    // Lightweight fenced-code parser: splits on ``` ... ``` blocks.
    // We intentionally avoid full Markdown parsing to keep the demo small.
    return androidx.compose.runtime.remember(text) {
        splitFencedCodeBlocks(text)
    }
}

private fun splitFencedCodeBlocks(input: String): List<MessagePart> {
    if (!input.contains("```")) return listOf(MessagePart.Text(input))

    val out = ArrayList<MessagePart>()
    var i = 0
    while (i < input.length) {
        val start = input.indexOf("```", startIndex = i)
        if (start < 0) {
            val tail = input.substring(i)
            if (tail.isNotEmpty()) out.add(MessagePart.Text(tail))
            break
        }
        if (start > i) {
            val before = input.substring(i, start)
            if (before.isNotEmpty()) out.add(MessagePart.Text(before))
        }
        val afterTicks = start + 3
        val end = input.indexOf("```", startIndex = afterTicks)
        if (end < 0) {
            // Unclosed fence: treat rest as text.
            out.add(MessagePart.Text(input.substring(start)))
            break
        }

        // Strip optional language hint on the first line (```lang).
        val raw = input.substring(afterTicks, end)
        val trimmedLeadingNewline = raw.removePrefix("\n").removePrefix("\r\n")
        val firstNl = trimmedLeadingNewline.indexOf('\n')
        val code = if (firstNl >= 0) {
            val firstLine = trimmedLeadingNewline.substring(0, firstNl)
            val looksLikeLang = firstLine.length <= 20 && firstLine.all { it.isLetterOrDigit() || it == '-' || it == '_' || it == '+' || it == '.' }
            if (looksLikeLang) trimmedLeadingNewline.substring(firstNl + 1) else trimmedLeadingNewline
        } else {
            trimmedLeadingNewline
        }.trimEnd()

        out.add(MessagePart.Code(code))
        i = end + 3
    }

    return out
}

