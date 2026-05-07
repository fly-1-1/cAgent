package com.example.cagent.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
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
                val prefix = when (msg.role) {
                    Role.User -> "You"
                    Role.Assistant -> "Assistant"
                    Role.System -> "System"
                }
                val display = if (msg.role == Role.Assistant && msg.text.isEmpty()) {
                    "$prefix: …"
                } else {
                    "$prefix: ${msg.text}"
                }
                Text(display)
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

